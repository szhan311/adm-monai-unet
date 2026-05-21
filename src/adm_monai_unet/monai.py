"""MONAI-style wrappers for the ADM timestep-conditioned U-Net."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from .adm_unet import UNetModel


def _as_tuple(value: int | Sequence[int] | str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        if not value.strip():
            return ()
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if isinstance(value, int):
        return (value,)
    return tuple(int(v) for v in value)


def _is_stride_two(stride: int | Sequence[int]) -> bool:
    if isinstance(stride, int):
        return stride == 2
    return all(int(v) == 2 for v in stride)


def _first_image_size(image_size: int | Sequence[int] | None) -> int:
    if image_size is None:
        return 0
    if isinstance(image_size, int):
        return image_size
    sizes = tuple(int(v) for v in image_size)
    if not sizes:
        return 0
    return sizes[0]


def _channel_mult_from_channels(
    channels: Sequence[int],
    model_channels: int | None,
    channel_mult: Sequence[int | float] | str | None,
) -> tuple[int | float, ...]:
    if channel_mult is not None:
        if isinstance(channel_mult, str):
            return tuple(
                float(v.strip()) if "." in v else int(v.strip())
                for v in channel_mult.split(",")
            )
        return tuple(channel_mult)

    if not channels:
        raise ValueError("channels must contain at least one level.")

    base = int(model_channels or channels[0])
    if base <= 0:
        raise ValueError("model_channels must be positive.")

    multipliers: list[int | float] = []
    for width in channels:
        ratio = int(width) / base
        if ratio.is_integer():
            multipliers.append(int(ratio))
        else:
            multipliers.append(ratio)
    return tuple(multipliers)


def _attention_resolutions_to_factors(
    *,
    image_size: int | Sequence[int] | None,
    attention_resolutions: int | Sequence[int] | str | None,
    attention_downsample_factors: int | Sequence[int] | str | None,
) -> tuple[int, ...]:
    if attention_downsample_factors is not None:
        return _as_tuple(attention_downsample_factors)

    resolutions = _as_tuple(attention_resolutions)
    if not resolutions:
        return ()

    size = _first_image_size(image_size)
    if size <= 0:
        return resolutions

    factors: list[int] = []
    for resolution in resolutions:
        if resolution <= 0:
            raise ValueError("attention_resolutions must be positive.")
        factors.append(size // resolution if size % resolution == 0 else resolution)
    return tuple(factors)


class ADMUNet(nn.Module):
    """ADM U-Net with a MONAI-compatible constructor and default ``forward(x)``.

    The wrapper intentionally accepts the most common ``monai.networks.nets.UNet``
    arguments so existing config files can switch the class name with minimal edits.
    MONAI-specific block choices such as ``act`` and ``norm`` are accepted for config
    compatibility; the underlying architecture remains the ADM residual U-Net.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128),
        strides: Sequence[int | Sequence[int]] | None = None,
        kernel_size: int | Sequence[int] = 3,
        up_kernel_size: int | Sequence[int] = 3,
        num_res_units: int = 2,
        act: str | tuple[Any, ...] | None = None,
        norm: str | tuple[Any, ...] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: int | None = None,
        image_size: int | Sequence[int] | None = None,
        model_channels: int | None = None,
        channel_mult: Sequence[int | float] | str | None = None,
        attention_resolutions: int | Sequence[int] | str | None = None,
        attention_downsample_factors: int | Sequence[int] | str | None = None,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = True,
        use_new_attention_order: bool = False,
        attention_type: str = "legacy",
        class_cond: bool = False,
        num_classes: int | None = None,
        condition_mode: str | None = None,
        default_timestep: float = 0.0,
        **unused_monai_kwargs: Any,
    ) -> None:
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions
        if spatial_dims not in (1, 2, 3):
            raise ValueError(f"spatial_dims must be 1, 2, or 3; got {spatial_dims}.")

        channels = tuple(int(c) for c in channels)
        if len(channels) == 0:
            raise ValueError("channels must contain at least one level.")
        if strides is not None:
            strides = tuple(strides)
            if len(strides) != max(0, len(channels) - 1):
                raise ValueError(
                    "strides must have len(channels) - 1 entries, matching MONAI UNet."
                )
            if not all(_is_stride_two(stride) for stride in strides):
                raise ValueError("ADMUNet currently supports stride-2 downsampling between levels.")

        if _as_tuple(kernel_size) not in {(3,), (3,) * spatial_dims}:
            raise ValueError("ADMUNet uses 3x3 convolutions; kernel_size must be 3.")
        if _as_tuple(up_kernel_size) not in {(3,), (3,) * spatial_dims}:
            raise ValueError("ADMUNet uses 3x3 upsampling convolutions; up_kernel_size must be 3.")

        base_channels = int(model_channels or channels[0])
        multipliers = _channel_mult_from_channels(channels, base_channels, channel_mult)
        attention_factors = _attention_resolutions_to_factors(
            image_size=image_size,
            attention_resolutions=attention_resolutions,
            attention_downsample_factors=attention_downsample_factors,
        )
        class_count = num_classes if class_cond else None

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.default_timestep = float(default_timestep)
        self.condition_mode = condition_mode
        self.unused_monai_kwargs = dict(unused_monai_kwargs)
        self.monai_compat = {
            "act": act,
            "norm": norm,
            "bias": bias,
            "adn_ordering": adn_ordering,
        }

        self.model = UNetModel(
            image_size=_first_image_size(image_size),
            in_channels=in_channels,
            model_channels=base_channels,
            out_channels=out_channels,
            num_res_blocks=max(1, int(num_res_units)),
            attention_resolutions=attention_factors,
            dropout=float(dropout),
            channel_mult=multipliers,
            dims=spatial_dims,
            num_classes=class_count,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            attention_type=attention_type,
            condition_mode=condition_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | float | int | None = None,
        xT: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the ADM U-Net.

        ``forward(x)`` works with MONAI trainers and inferers. Diffusion code can
        still pass explicit ``timesteps``, ``xT`` and class labels.
        """
        if self.condition_mode == "concat" and xT is None:
            raise ValueError("condition_mode='concat' requires xT in forward().")
        if timesteps is None:
            timesteps = torch.full(
                (x.shape[0],),
                self.default_timestep,
                device=x.device,
                dtype=torch.float32,
            )
        elif isinstance(timesteps, (int, float)):
            timesteps = torch.full((x.shape[0],), float(timesteps), device=x.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps.to(device=x.device).expand(x.shape[0])
        else:
            timesteps = timesteps.to(device=x.device)
        return self.model(x, timesteps=timesteps, xT=xT, y=y)


class ADMUNet3D(ADMUNet):
    """3D-specialized alias for MONAI projects that prefer explicit class names."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("spatial_dims", 3)
        super().__init__(*args, **kwargs)


def create_adm_unet_from_monai_kwargs(**kwargs: Any) -> ADMUNet:
    """Create an :class:`ADMUNet` from a MONAI-style UNet kwargs dictionary."""
    return ADMUNet(**kwargs)


# Exact-spelling aliases for projects that refer to the architecture as admUNet.
admUNet = ADMUNet
admUNet3D = ADMUNet3D
