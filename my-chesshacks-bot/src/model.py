import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ChessModel(nn.Module):
    def __init__(self, num_res_blocks=6, num_channels=256):
        super(ChessModel, self).__init__()

        self.conv_input = nn.Conv2d(
            12, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(num_channels)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        self.policy_conv = nn.Conv2d(num_channels, 128, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(128)
        self.policy_fc = nn.Linear(128 * 8 * 8, 4096)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))

        for res_block in self.res_blocks:
            x = res_block(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        return policy
