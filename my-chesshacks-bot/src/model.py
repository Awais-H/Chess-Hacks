import torch
import torch.nn as nn
import torch.nn.functional as F

# define model architecture

class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(0.2)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out

class ChessModel(nn.Module):
    def __init__(self, num_res_blocks=12, num_filters=256, input_channels=18):
        super().__init__()
        
        # Initial convolution (now accepts 18 channels)
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Residual tower (increased from 6 to 12)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy head - predicts moves
        self.policy_conv = nn.Conv2d(num_filters, 128, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)  # Regularization
        self.policy_fc = nn.Linear(128 * 8 * 8, 4096)
    
    def forward(self, x):
        # Input processing
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.dropout(policy)
        output = self.policy_fc(policy)
        
        return output