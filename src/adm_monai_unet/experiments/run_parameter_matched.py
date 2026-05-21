"""Parameter-matched ADMUNet vs MONAI UNet comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .run_comparison import TASKS, _train_one

MATCHED_CONFIGS = (
    {
        "label": "adm_unet_c8_16_ru1",
        "model_name": "adm_unet",
        "channels": (8, 16),
        "num_res_units": 1,
    },
    {
        "label": "monai_unet_c8_16_32_64_ru0",
        "model_name": "monai_unet",
        "channels": (8, 16, 32, 64),
        "num_res_units": 0,
    },
    {
        "label": "monai_unet_c12_24_24_ru3",
        "model_name": "monai_unet",
        "channels": (12, 24, 24),
        "num_res_units": 3,
    },
    {
        "label": "monai_unet_c25_50_ru2",
        "model_name": "monai_unet",
        "channels": (25, 50),
        "num_res_units": 2,
    },
    {
        "label": "monai_unet_c38_114_ru0",
        "model_name": "monai_unet",
        "channels": (38, 114),
        "num_res_units": 0,
    },
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/param_matched"))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--spatial-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    rows = []
    for task in TASKS:
        for config in MATCHED_CONFIGS:
            row = _train_one(
                config["model_name"],
                task,
                channels=config["channels"],
                num_res_units=config["num_res_units"],
                spatial_size=args.spatial_size,
                steps=args.steps,
                batch_size=args.batch_size,
                eval_batches=args.eval_batches,
                lr=args.lr,
                device=device,
                seed=args.seed,
            )
            row["label"] = config["label"]
            row["channels"] = ",".join(str(v) for v in config["channels"])
            row["num_res_units"] = config["num_res_units"]
            rows.append(row)
            print(json.dumps(row, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
