"""Small representative comparisons between MONAI UNet and ADMUNet.

The tasks are synthetic but shaped like common 3D medical-imaging problems:
segmentation, denoising, and paired contrast translation. They are intentionally
small enough to run on CPU for API and regression checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adm_monai_unet import ADMUNet


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str
    higher_is_better: bool
    metric_name: str


TASKS = (
    TaskSpec("sphere_segmentation", "segmentation", True, "dice"),
    TaskSpec("bias_field_denoising", "regression", False, "mse"),
    TaskSpec("contrast_translation", "regression", False, "mse"),
)


def _load_monai_unet():
    try:
        from monai.networks.nets import UNet
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MONAI is required for the baseline. Install with "
            "`pip install adm-monai-unet[experiments]` or `pip install monai`."
        ) from exc
    return UNet


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _grid(spatial_size: int, device: torch.device) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, spatial_size, device=device)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([zz, yy, xx], dim=0)


def _random_blob(
    *,
    batch_size: int,
    spatial_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid = _grid(spatial_size, device).unsqueeze(0)
    center = torch.empty(batch_size, 3, 1, 1, 1, device=device).uniform_(
        -0.35,
        0.35,
        generator=generator,
    )
    radii = torch.empty(batch_size, 1, 1, 1, 1, device=device).uniform_(
        0.28,
        0.48,
        generator=generator,
    )
    anisotropy = torch.empty(batch_size, 3, 1, 1, 1, device=device).uniform_(
        0.85,
        1.25,
        generator=generator,
    )
    normalized = ((grid - center) / anisotropy).pow(2).sum(dim=1, keepdim=True).sqrt()
    mask = (normalized < radii).float()
    soft = torch.exp(-0.5 * (normalized / radii.clamp_min(1e-3)).pow(2))
    return grid, mask, soft


def _make_batch(
    task_name: str,
    *,
    batch_size: int,
    spatial_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid, mask, soft = _random_blob(
        batch_size=batch_size,
        spatial_size=spatial_size,
        device=device,
        generator=generator,
    )
    noise = torch.randn(
        batch_size,
        1,
        spatial_size,
        spatial_size,
        spatial_size,
        device=device,
        generator=generator,
    )
    bias = 1.0 + 0.35 * grid[:, :1] - 0.25 * grid[:, 1:2]

    if task_name == "sphere_segmentation":
        x = mask * bias + 0.30 * noise + 0.10 * soft
        return x.float(), mask.float()

    if task_name == "bias_field_denoising":
        clean = (0.25 + 0.75 * soft) * mask
        corrupted = clean * bias + 0.20 * noise
        return corrupted.float(), clean.float()

    if task_name == "contrast_translation":
        source = (0.20 + 0.80 * soft) * bias + 0.18 * noise
        target = (mask * (1.0 - soft) + 0.65 * soft) * (1.0 + 0.25 * grid[:, 2:3])
        return source.float(), target.float()

    raise ValueError(f"Unknown task: {task_name}")


def _build_model(
    model_name: str,
    *,
    channels: tuple[int, ...],
    num_res_units: int,
    spatial_size: int,
    device: torch.device,
) -> torch.nn.Module:
    if model_name == "monai_unet":
        UNet = _load_monai_unet()
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=channels,
            strides=(2,) * (len(channels) - 1),
            num_res_units=num_res_units,
        )
    elif model_name == "adm_unet":
        model = ADMUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=channels,
            strides=(2,) * (len(channels) - 1),
            num_res_units=num_res_units,
            image_size=spatial_size,
            attention_resolutions=(),
            attention_type="legacy",
            default_timestep=0.0,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def _loss(task: TaskSpec, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if task.kind == "segmentation":
        return F.binary_cross_entropy_with_logits(pred, target)
    return F.mse_loss(pred, target)


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    task: TaskSpec,
    *,
    batches: int,
    batch_size: int,
    spatial_size: int,
    device: torch.device,
    seed: int,
) -> float:
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    values: list[float] = []
    for _ in range(batches):
        x, target = _make_batch(
            task.name,
            batch_size=batch_size,
            spatial_size=spatial_size,
            device=device,
            generator=generator,
        )
        pred = model(x)
        if task.kind == "segmentation":
            pred_mask = (torch.sigmoid(pred) > 0.5).float()
            intersection = (pred_mask * target).sum(dim=(1, 2, 3, 4))
            denom = pred_mask.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
            dice = ((2.0 * intersection + 1e-6) / (denom + 1e-6)).mean()
            values.append(float(dice.cpu()))
        else:
            values.append(float(F.mse_loss(pred, target).cpu()))
    return float(np.mean(values))


def _train_one(
    model_name: str,
    task: TaskSpec,
    *,
    channels: tuple[int, ...],
    num_res_units: int,
    spatial_size: int,
    steps: int,
    batch_size: int,
    eval_batches: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> dict[str, float | int | str]:
    _seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    model = _build_model(
        model_name,
        channels=channels,
        num_res_units=num_res_units,
        spatial_size=spatial_size,
        device=device,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    start = time.perf_counter()
    model.train()
    last_loss = math.nan
    for _ in range(steps):
        x, target = _make_batch(
            task.name,
            batch_size=batch_size,
            spatial_size=spatial_size,
            device=device,
            generator=generator,
        )
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = _loss(task, pred, target)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
    train_seconds = time.perf_counter() - start

    metric = _evaluate(
        model,
        task,
        batches=eval_batches,
        batch_size=batch_size,
        spatial_size=spatial_size,
        device=device,
        seed=seed + 101,
    )
    return {
        "task": task.name,
        "model": model_name,
        "metric_name": task.metric_name,
        "metric": metric,
        "higher_is_better": task.higher_is_better,
        "last_train_loss": last_loss,
        "train_seconds": train_seconds,
        "parameters": _count_parameters(model),
    }


def _parse_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(channels) < 2:
        raise argparse.ArgumentTypeError("channels must include at least two levels.")
    return channels


def _write_results(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--spatial-size", type=int, default=16)
    parser.add_argument("--channels", type=_parse_channels, default=(8, 16))
    parser.add_argument("--num-res-units", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    rows: list[dict[str, float | int | str]] = []
    for task in TASKS:
        for model_name in ("monai_unet", "adm_unet"):
            rows.append(
                _train_one(
                    model_name,
                    task,
                    channels=args.channels,
                    num_res_units=args.num_res_units,
                    spatial_size=args.spatial_size,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    eval_batches=args.eval_batches,
                    lr=args.lr,
                    device=device,
                    seed=args.seed,
                )
            )

    _write_results(rows, args.output_dir)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
