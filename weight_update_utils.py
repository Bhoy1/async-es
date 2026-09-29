
from collections.abc import Iterable

import torch


def generate_noise(
    seed: int,
    shape: torch.Size | tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return torch.randn(shape, dtype=dtype, device=device, generator=gen)


@torch.inference_mode()
def apply_seeded_noise_to_parameters(
    parameters: Iterable[torch.nn.Parameter],
    seed: int,
    scale: float,
) -> None:
    seed = int(seed)
    scale = float(scale)

    for parameter in parameters:
        noise = generate_noise(
            seed,
            parameter.shape,
            parameter.dtype,
            parameter.device,
        )
        parameter.add_(noise, alpha=scale)


@torch.inference_mode()
def update_parameters_from_seeds(
    parameters: Iterable[torch.nn.Parameter],
    seeds: list[int],
    coeffs: list[float],
    alpha: float,
    population_size: int,
) -> None:
    if len(seeds) != len(coeffs):
        raise ValueError("seeds and coeffs must have equal length")

    seeds = [int(seed) for seed in seeds]
    coeffs = [float(coeff) for coeff in coeffs]
    scale = float(alpha) / float(population_size)

    for parameter in parameters:
        accumulator = torch.zeros_like(parameter)
        for seed, coeff in zip(seeds, coeffs):
            noise = generate_noise(
                seed,
                parameter.shape,
                parameter.dtype,
                parameter.device,
            )
            accumulator.add_(noise, alpha=coeff)

        parameter.add_(accumulator.to(dtype=parameter.dtype), alpha=scale)
