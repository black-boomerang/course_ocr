from pytorch_metric_learning.losses import ArcFaceLoss
from torch import nn
from torch.nn import functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(n_channels, n_channels, kernel_size=7, padding=3, groups=n_channels)
        self.norm = nn.LayerNorm(n_channels, eps=1e-6)
        self.pw_conv1 = nn.Linear(n_channels, 4 * n_channels)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * n_channels, n_channels)

    def forward(self, x):
        input = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + x
        return x


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNeXtWithArcFace(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super(ConvNeXtWithArcFace, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            ConvNeXtBlock(16),
            LayerNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),
            ConvNeXtBlock(32),
            LayerNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            ConvNeXtBlock(64),
            LayerNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            ConvNeXtBlock(128),
            LayerNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            ConvNeXtBlock(256),
            nn.Flatten(),
            nn.Linear(5 * 5 * 256, embed_dim)
        )
        self.loss_fn = ArcFaceLoss(margin=28.6, scale=16, num_classes=num_classes, embedding_size=embed_dim)

    def forward(self, x):
        return self.layers(x)


class LeNetWithArcFace(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super(LeNetWithArcFace, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2)),

            nn.Flatten(),
            nn.Linear(5 * 5 * 256, embed_dim)
        )
        self.loss_fn = ArcFaceLoss(margin=28.6, scale=16, num_classes=num_classes, embedding_size=embed_dim)

    def forward(self, x):
        return self.layers(x)
