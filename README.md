# adm-monai-unet

[![CI](https://github.com/szhan311/adm-monai-unet/actions/workflows/ci.yml/badge.svg)](https://github.com/szhan311/adm-monai-unet/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/adm-monai-unet.svg)](https://pypi.org/project/adm-monai-unet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`adm-monai-unet` packages the ADM timestep-conditioned 3D U-Net from
`sdb-brain/utils/unet3d.py` behind a MONAI-style interface.

The main entry point is `adm_monai_unet.ADMUNet`. It accepts the common
`monai.networks.nets.UNet` constructor arguments (`spatial_dims`, `in_channels`,
`out_channels`, `channels`, `strides`, `num_res_units`, `dropout`) while keeping
the ADM residual blocks, timestep embedding, and attention blocks.

## Install

From PyPI after release:

```bash
pip install adm-monai-unet
```

From GitHub:

```bash
pip install git+ssh://git@github.com/szhan311/adm-monai-unet.git
```

For local development:

```bash
git clone git@github.com:szhan311/adm-monai-unet.git
cd adm-monai-unet
pip install -e ".[dev,experiments]"
```

`flash-attn` is optional. The packaged wrapper defaults to legacy attention so a
plain CPU or CUDA PyTorch environment can import and run it.

## MONAI-Style Usage

```python
import torch
from adm_monai_unet import ADMUNet

model = ADMUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64),
    strides=(2, 2),
    num_res_units=1,
    image_size=32,
)

x = torch.randn(1, 1, 32, 32, 32)
y = model(x)  # MONAI trainer/inferer compatible forward(x)
```

Diffusion-style calls are still supported:

```python
timesteps = torch.tensor([100.0])
y = model(x, timesteps=timesteps)
```

## Notes On Compatibility

- `ADMUNet` supports `spatial_dims` 1, 2, and 3; this package is primarily meant
  for the 3D case.
- Downsampling currently follows the ADM implementation and expects stride-2
  transitions between channel levels.
- MONAI block-selection options like `act`, `norm`, `bias`, and `adn_ordering`
  are accepted so existing config dictionaries do not break, but the underlying
  architecture remains ADM.
- `attention_resolutions` follows the ADM convention: pass spatial resolutions
  such as `(16, 8)` together with `image_size=128`, or pass
  `attention_downsample_factors` directly.

## Representative Experiments

Run the small reproducible comparison suite:

```bash
adm-unet-experiments --output-dir results/local/quick --steps 60 --spatial-size 16
```

or:

```bash
python scripts/run_representative_experiments.py --output-dir results/local/quick
```

The suite compares `monai.networks.nets.UNet` and `ADMUNet` on three small 3D
tasks: sphere segmentation, bias-field denoising, and paired contrast
translation. Results are written to `summary.json` and `summary.csv`.

Quick CPU run in this environment:

| Task | Metric | MONAI UNet | ADMUNet | Better |
| --- | ---: | ---: | ---: | --- |
| Sphere segmentation | Dice, higher is better | 0.3520 | 0.9317 | ADMUNet |
| Bias-field denoising | MSE, lower is better | 0.0255 | 0.0033 | ADMUNet |
| Contrast translation | MSE, lower is better | 0.0266 | 0.0032 | ADMUNet |

Run settings: `--steps 60 --eval-batches 8 --batch-size 2 --spatial-size 16
--channels 8,16 --threads 4`, MONAI 1.5.2, PyTorch 2.8.0 CPU. The ADM model in
this tiny setup has more parameters (122,489 vs 4,744), so these results support
the model-quality claim but are not a parameter-matched benchmark.

## Deployment

For training or inference environments, install the package into the same
environment as PyTorch and MONAI:

```bash
pip install "adm-monai-unet[monai]"
```

Docker image for a quick smoke run:

```bash
docker build -t adm-monai-unet:latest .
docker run --rm adm-monai-unet:latest
```

## Release

The repository includes:

- GitHub Actions CI in `.github/workflows/ci.yml`
- PyPI trusted-publishing workflow in `.github/workflows/publish.yml`
- release checklist in `docs/release.md`

To publish, configure PyPI trusted publishing for this repository, tag a
release, and publish the GitHub release. The `Publish` workflow will build and
upload the wheel and source distribution.

## License And Attribution

This project is released under the MIT License. The vendored ADM-style model
code follows OpenAI `guided-diffusion`, which is also MIT licensed. See
`THIRD_PARTY_NOTICES.md` for the upstream notice.
