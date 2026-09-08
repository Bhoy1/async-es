"""Persistent LoRA state and deterministic ES operations.

The central adapter lives on CPU. Each rollout receives an immutable PEFT
adapter snapshot, which keeps asynchronous evaluations tied to their dispatch
policy without copying or modifying the frozen base model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable

import torch
from safetensors.torch import load_file, save_file


LORA_COMPONENTS = ("B", "A")


@dataclass(frozen=True)
class LoraSpec:
    rank: int
    lora_alpha: int
    target_modules: tuple[str, ...]

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.rank


def component_for_version(version: int) -> str:
    """Use the effective B factor first, then alternate B/A by update."""
    return LORA_COMPONENTS[int(version) % len(LORA_COMPONENTS)]


def _component_for_key(key: str) -> str | None:
    if ".lora_A." in key or key.endswith(".lora_A.weight"):
        return "A"
    if ".lora_B." in key or key.endswith(".lora_B.weight"):
        return "B"
    return None


def _adapter_state_from_meta_model(
    model_name: str,
    spec: LoraSpec,
    initialization_seed: int,
) -> dict[str, torch.Tensor]:
    """Construct PEFT-compatible adapter tensors without allocating base weights."""
    from accelerate import init_empty_weights
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
    from transformers import AutoConfig, AutoModelForCausalLM

    model_config = AutoConfig.from_pretrained(model_name)
    peft_config = LoraConfig(
        r=spec.rank,
        lora_alpha=spec.lora_alpha,
        lora_dropout=0.0,
        target_modules=list(spec.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
    )
    peft_config.base_model_name_or_path = model_name
    with init_empty_weights():
        base_model = AutoModelForCausalLM.from_config(model_config)
        peft_model = get_peft_model(base_model, peft_config)
    meta_state = get_peft_model_state_dict(peft_model)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(initialization_seed))
    state: dict[str, torch.Tensor] = {}
    for key, value in sorted(meta_state.items()):
        component = _component_for_key(key)
        if component is None:
            continue
        tensor = torch.empty(tuple(value.shape), dtype=torch.float32, device="cpu")
        if component == "A":
            torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), generator=generator)
        else:
            tensor.zero_()
        state[key] = tensor
    if not state or not any(_component_for_key(key) == "A" for key in state):
        raise RuntimeError("PEFT did not create LoRA A/B tensors for the requested modules")
    return state


def _write_adapter_config(path: Path, model_name: str, spec: LoraSpec) -> None:
    from peft import LoraConfig, TaskType

    config = LoraConfig(
        r=spec.rank,
        lora_alpha=spec.lora_alpha,
        lora_dropout=0.0,
        target_modules=list(spec.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
    )
    config.base_model_name_or_path = model_name
    config.save_pretrained(path)


class LoraPolicyState:
    """CPU-resident persistent LoRA adapter optimized by alternating ES updates."""

    def __init__(
        self,
        *,
        model_name: str,
        spec: LoraSpec,
        state: dict[str, torch.Tensor],
    ) -> None:
        self.model_name = model_name
        self.spec = spec
        self.state = {
            key: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for key, value in sorted(state.items())
        }
        self._validate_state()

    @classmethod
    def initialize(
        cls,
        model_name: str,
        spec: LoraSpec,
        initialization_seed: int = 0,
    ) -> "LoraPolicyState":
        return cls(
            model_name=model_name,
            spec=spec,
            state=_adapter_state_from_meta_model(
                model_name, spec, initialization_seed
            ),
        )

    @classmethod
    def load(cls, path: str | Path, expected_spec: LoraSpec) -> "LoraPolicyState":
        path = Path(path)
        config = json.loads((path / "adapter_config.json").read_text())
        actual_spec = LoraSpec(
            rank=int(config["r"]),
            lora_alpha=int(config["lora_alpha"]),
            target_modules=tuple(sorted(config["target_modules"])),
        )
        normalized_expected = LoraSpec(
            rank=expected_spec.rank,
            lora_alpha=expected_spec.lora_alpha,
            target_modules=tuple(sorted(expected_spec.target_modules)),
        )
        if actual_spec != normalized_expected:
            raise ValueError(
                f"LoRA checkpoint configuration mismatch: {actual_spec} != "
                f"{normalized_expected}"
            )
        model_name = str(config.get("base_model_name_or_path") or "")
        return cls(
            model_name=model_name,
            spec=expected_spec,
            state=load_file(path / "adapter_model.safetensors", device="cpu"),
        )

    def _validate_state(self) -> None:
        components = {_component_for_key(key) for key in self.state}
        if not {"A", "B"}.issubset(components):
            raise ValueError("LoRA state must contain both A and B factor tensors")
        unknown = [key for key in self.state if _component_for_key(key) is None]
        if unknown:
            raise ValueError(f"Unsupported non-LoRA tensors in adapter state: {unknown}")

    def keys_for_component(self, component: str) -> list[str]:
        if component not in LORA_COMPONENTS:
            raise ValueError(f"Unknown LoRA component: {component!r}")
        return [key for key in self.state if _component_for_key(key) == component]

    def _noise(self, seed: int, component: str) -> Iterable[tuple[str, torch.Tensor]]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        for key in self.keys_for_component(component):
            yield key, torch.randn(
                self.state[key].shape,
                dtype=self.state[key].dtype,
                device="cpu",
                generator=generator,
            )

    @torch.inference_mode()
    def apply_update(
        self,
        *,
        seeds: list[int],
        coefficients: list[float],
        alpha: float,
        population_size: int,
        component: str,
    ) -> None:
        if len(seeds) != len(coefficients):
            raise ValueError("seeds and coefficients must have equal length")
        if len(seeds) != population_size:
            raise ValueError("LoRA update cohort must equal population_size")
        accumulators = {
            key: torch.zeros_like(self.state[key])
            for key in self.keys_for_component(component)
        }
        for seed, coefficient in zip(seeds, coefficients):
            for key, noise in self._noise(seed, component):
                accumulators[key].add_(noise, alpha=float(coefficient))
        scale = float(alpha) / float(population_size)
        for key, update in accumulators.items():
            self.state[key].add_(update, alpha=scale)

    def candidate_state(
        self, *, seed: int, sigma: float, component: str
    ) -> dict[str, torch.Tensor]:
        candidate = {key: value.clone() for key, value in self.state.items()}
        for key, noise in self._noise(seed, component):
            candidate[key].add_(noise, alpha=float(sigma))
        return candidate

    def save(
        self,
        path: str | Path,
        *,
        state: dict[str, torch.Tensor] | None = None,
        trainer_state: dict[str, Any] | None = None,
    ) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        _write_adapter_config(path, self.model_name, self.spec)
        tensors = self.state if state is None else state
        save_file(
            {key: value.contiguous() for key, value in tensors.items()},
            path / "adapter_model.safetensors",
            metadata={"format": "pt"},
        )
        if trainer_state is not None:
            (path / "trainer_state.json").write_text(
                json.dumps(trainer_state, indent=2, default=str) + "\n"
            )
        return path

    def save_candidate(
        self,
        path: str | Path,
        *,
        seed: int,
        sigma: float,
        component: str,
    ) -> Path:
        return self.save(
            path,
            state=self.candidate_state(seed=seed, sigma=sigma, component=component),
        )

    @staticmethod
    def remove_snapshot(path: str | Path | None) -> None:
        if path:
            shutil.rmtree(path, ignore_errors=True)

    def stats(self) -> dict[str, Any]:
        component_counts = {
            component: sum(
                self.state[key].numel()
                for key in self.keys_for_component(component)
            )
            for component in LORA_COMPONENTS
        }
        return {
            "parameterization": "lora",
            "rank": self.spec.rank,
            "lora_alpha": self.spec.lora_alpha,
            "lora_scaling": self.spec.scaling,
            "lora_dropout": 0.0,
            "target_modules": list(self.spec.target_modules),
            "component_order": list(LORA_COMPONENTS),
            "active_parameters": sum(component_counts.values()),
            "component_parameters": component_counts,
        }
