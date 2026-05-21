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

volume = torch.randn(1, 1, 32, 32, 32)
logits = model(volume)
print(logits.shape)

# Diffusion-style use is still available when explicit timesteps are needed.
timesteps = torch.tensor([100.0])
logits_at_t = model(volume, timesteps=timesteps)
print(logits_at_t.shape)
