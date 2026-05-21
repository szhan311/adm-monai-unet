import torch

from adm_monai_unet import ADMUNet

model = ADMUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32),
    strides=(2,),
    num_res_units=1,
    image_size=32,
    class_cond=True,
    num_classes=4,
)

volume = torch.randn(2, 1, 32, 32, 32)
condition = torch.tensor([0, 3], dtype=torch.long)
logits = model(volume, y=condition)
print(logits.shape)
