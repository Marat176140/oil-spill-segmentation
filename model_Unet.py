import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder.append(double_conv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = double_conv(features[-1], features[-1] * 2)

        # Decoder
        self.decoder_up   = nn.ModuleList()
        self.decoder_conv = nn.ModuleList()

        for feature in reversed(features):
            self.decoder_up.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_conv.append(
                double_conv(feature * 2, feature)
            )

        # Output layer
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]  # разворачиваем

        for i in range(len(self.decoder_up)):
            x = self.decoder_up[i](x)
            skip = skip_connections[i]
            x = torch.cat([skip, x], dim=1)  # skip connection
            x = self.decoder_conv[i](x)

        return self.output(x)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(8, 3, 256, 256) 
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")  # [8, 1, 256, 256]
    print(f"Parametry modelu: {sum(p.numel() for p in model.parameters()):,}")

