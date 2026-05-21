import pytest
import torch

from adm_monai_unet import ADMUNet, ADMUNet3D, admUNet


def test_forward_monai_style_3d_shape():
    model = ADMUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(8, 16),
        strides=(2,),
        num_res_units=1,
        image_size=16,
    )
    x = torch.randn(2, 1, 16, 16, 16)
    y = model(x)
    assert y.shape == (2, 2, 16, 16, 16)


def test_explicit_timesteps_and_aliases():
    assert admUNet is ADMUNet
    model = ADMUNet3D(
        in_channels=1,
        out_channels=1,
        channels=(8, 16),
        strides=(2,),
        num_res_units=1,
        image_size=16,
    )
    x = torch.randn(1, 1, 16, 16, 16)
    y = model(x, timesteps=torch.tensor([10.0]))
    assert y.shape == x.shape


def test_class_conditional_forward_requires_labels():
    model = ADMUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(8, 16),
        strides=(2,),
        num_res_units=1,
        image_size=16,
        class_cond=True,
        num_classes=3,
    )
    x = torch.randn(2, 1, 16, 16, 16)
    labels = torch.tensor([0, 2], dtype=torch.long)
    y = model(x, y=labels)
    assert y.shape == (2, 2, 16, 16, 16)

    with pytest.raises(AssertionError):
        model(x)
