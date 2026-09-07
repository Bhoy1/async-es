# Built on es-at-scale; low-level utilities also adapt code from es-awd.
# Modified for bounded-staleness ES, scoped perturbations, modern vLLM,
# and portable Ray startup (2026-09).
import os
import tempfile
import torch

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM
# Try importing get_ip and get_open_port from the new location first, then fall back to the old location for backward compatibility
try:
    from vllm.utils.network_utils import get_ip, get_open_port
except ImportError:
    from vllm.utils import get_ip, get_open_port
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from weight_update_utils import (
    apply_seeded_noise_to_parameters,
    select_parameter_references,
    select_parameters,
    summarize_parameter_scope,
    update_parameters_from_seeds,
)


class ESNcclLLM(LLM):
    """
    Custom LLM class that extends vLLM's LLM to include methods for weight
    perturbation, restoration, and inter-engine communication using NCCL.
    This class is designed to be used as a Ray actor for distributed ES training.
    """
    def __init__(self, *args, **kwargs):
        model_name = kwargs.get("model")
        self._cached_repr = f"ESNcclLLM(model={model_name!r}, status='initializing')"
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Avoid adding non-serializable objects; rely on worker-side ops
        super().__init__(*args, **kwargs)
        model_config = getattr(self, "model_config", None)
        if model_config is None and hasattr(self, "llm_engine"):
            model_config = getattr(self.llm_engine, "model_config", None)
        resolved_model_name = getattr(model_config, "model", model_name)
        self._cached_repr = f"ESNcclLLM(model={resolved_model_name!r})"

    def __repr__(self) -> str:
        return self._cached_repr

def launch_engines(
    num_engines,
    model_name,
    precision="bfloat16",
    max_model_len=None,
    gpu_memory_utilization=0.9,
):

    # Clean up any existing Ray state from the environment to avoid conflicts with previous runs
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)

    # Ray places long session/socket names below this directory. Keep the
    # generated prefix short so AF_UNIX paths remain below Linux's 107-byte
    # limit even when TMPDIR is used separately by vLLM.
    ray_tmp_root = os.environ.get("RAY_TMPDIR")
    if ray_tmp_root:
        os.makedirs(ray_tmp_root, exist_ok=True)
    unique_dir = tempfile.mkdtemp(prefix="r", dir=ray_tmp_root)
    # initialize Ray with the unique temporary directory and disable the dashboard to avoid port conflicts
    ray.init(
        address="local",
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=unique_dir,
        dashboard_port=None
    )

    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]
    enforce_eager = os.environ.get("ES_VLLM_ENFORCE_EAGER", "0").lower() in {"1", "true", "yes"}

    engine_kwargs = {
        "model": model_name,
        "tensor_parallel_size": 1,
        "distributed_executor_backend": "ray",
        "worker_extension_cls": "distributed_utils.WorkerExtension",
        "dtype": precision,
        "enable_prefix_caching": False,
        "enforce_eager": enforce_eager,
        "gpu_memory_utilization": float(gpu_memory_utilization),
    }
    if max_model_len is not None:
        engine_kwargs["max_model_len"] = int(max_model_len)

    engines = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            **engine_kwargs,
        )
        for strategy in strategies
    ]

    # initialize inter-engine communication groups
    master_address = get_ip()
    master_port = get_open_port()
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, num_engines)
        )
        for i in range(num_engines)
    ])

    return engines, pgs

def cleanup(engines, pgs, wandb_run):
        for llm in engines:
            try: ray.kill(llm)
            except: pass
        for pg in pgs:
            try: remove_placement_group(pg)
            except: pass
        try:
            wandb_run.finish()
        except Exception:
            pass
        ray.shutdown()


class WorkerExtension:
    """
    Methods used by the ES trainer:
    - perturb_self_weights: add noise to the worker's own weights based on a seed and scale
    - restore_self_weights: remove noise from worker's own weights based on a seed and scale
    - update_weights_from_seeds: update the worker's weights based on a set of seeds and coefficients
    - init_inter_engine_group: initialize the inter-engine communication group
    - broadcast_all_weights: broadcast all weights from one worker to all other workers in the inter-engine group
    """

    def _parameters_for_scope(self, perturbation_scope="all"):
        return select_parameters(
            self.model_runner.model.parameters(),
            perturbation_scope,
        )

    def perturb_self_weights(
        self,
        seed,
        noise_scale,
        caching=False,
        negate=False,
        perturbation_scope="all",
    ):
        apply_seeded_noise_to_parameters(
            self._parameters_for_scope(perturbation_scope),
            seed=seed,
            scale=-float(noise_scale) if negate else float(noise_scale),
            caching=caching,
        )
        return True

    def restore_self_weights(
        self,
        seed,
        noise_scale,
        caching=False,
        negate=False,
        perturbation_scope="all",
    ):
        apply_seeded_noise_to_parameters(
            self._parameters_for_scope(perturbation_scope),
            seed=seed,
            scale=float(noise_scale) if negate else -float(noise_scale),
            caching=caching,
        )
        return True

    def apply_random_walk_step(
        self,
        seed,
        step_size,
        perturbation_scope="matrix",
    ):
        """Permanently apply one reward-free Gaussian step."""
        apply_seeded_noise_to_parameters(
            self._parameters_for_scope(perturbation_scope),
            seed=seed,
            scale=float(step_size),
            caching=False,
        )
        return True

    def update_weights_from_seeds(
        self,
        seeds,
        coeffs,
        alpha,
        population_size,
        caching=False,
        weight_decay_type="none",
        weight_decay_lambda=0.0,
        perturbation_scope="all",
    ):
        original_parameters = None
        if getattr(self, "cpu_weight_snapshot", None) is not None:
            parameters, original_parameters = select_parameter_references(
                self.model_runner.model.parameters(),
                self.cpu_weight_snapshot,
                perturbation_scope,
            )
        else:
            parameters = self._parameters_for_scope(perturbation_scope)
        update_parameters_from_seeds(
            parameters,
            seeds=seeds,
            coeffs=coeffs,
            alpha=alpha,
            population_size=population_size,
            caching=caching,
            original_parameters=original_parameters,
            weight_decay_type=weight_decay_type,
            weight_decay_lambda=weight_decay_lambda,
        )
        return True

    def get_perturbation_scope_stats(self, perturbation_scope="all"):
        return summarize_parameter_scope(
            self.model_runner.model.named_parameters(),
            perturbation_scope,
        )

    @torch.inference_mode()
    def snapshot_weights_to_cpu(self):
        self.cpu_weight_snapshot = []
        for p in self.model_runner.model.parameters():
            cpu_copy = p.detach().to(device="cpu", copy=True)
            if torch.cuda.is_available():
                cpu_copy = cpu_copy.pin_memory()
            self.cpu_weight_snapshot.append(cpu_copy)
        return True

    @torch.inference_mode()
    def restore_self_weights_from_cpu(self):
        if not hasattr(self, "cpu_weight_snapshot"):
            raise RuntimeError("CPU weight snapshot is not initialized. Call snapshot_weights_to_cpu first.")

        for p, cpu_p in zip(self.model_runner.model.parameters(), self.cpu_weight_snapshot):
            p.copy_(cpu_p, non_blocking=cpu_p.is_pinned())

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        return True

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        pg = StatelessProcessGroup.create(
            host=master_address, port=master_port, rank=rank, world_size=world_size
        )
        self.inter_pg = PyNcclCommunicator(pg, device=self.device)
        return True

    def broadcast_all_weights(self, src_rank: int):
        for p in self.model_runner.model.parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        return True

    def write_weights_to_disk(self, path):
        torch.save(self.model_runner.model.state_dict(), path)
        return True

    @torch.inference_mode()
    def load_weights_from_disk(self, path):
        # Loading directly onto an initialized vLLM GPU temporarily duplicates
        # the full model and can OOM. Keep the checkpoint on CPU while
        # load_state_dict streams each tensor into the existing parameters.
        state_dict = torch.load(path, map_location="cpu")
        incompatible = self.model_runner.model.load_state_dict(state_dict, strict=False)
        del state_dict
        runtime_scale_suffixes = (
            ".attn._k_scale",
            ".attn._v_scale",
            ".attn._q_scale",
            ".attn._prob_scale",
        )
        missing_parameters = [
            key
            for key in incompatible.missing_keys
            if not key.endswith(runtime_scale_suffixes)
        ]
        if missing_parameters or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint is incompatible with the current vLLM model: "
                f"missing={missing_parameters}, unexpected={incompatible.unexpected_keys}"
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        return True
