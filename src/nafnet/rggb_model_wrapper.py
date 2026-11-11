import torch.nn as nn


class RGGBModelWrapper(nn.Module):
    def __init__(
        self, model: nn.Module, in_channels: int, out_channels_4x: int, width: int
    ):
        super().__init__()

        self.model = model

        self.intro = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=out_channels_4x,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        pixel_shuffle_factor = int((out_channels_4x / 3) ** 0.5)
        self.pixel_shuffle = nn.PixelShuffle(pixel_shuffle_factor)

    def forward(self, x):
        x = self.intro(x)
        x = self.model(x)
        x = self.ending(x)
        x = self.pixel_shuffle(x)
        return x
