"""
Improved chess model with piece value awareness and better architecture.
"""
import torch
import torch.nn as nn


class ChessModelV1(nn.Module):
    """
    Original simple CNN model (12 input channels - pieces only).
    Kept for backward compatibility with old checkpoints.
    """
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces 8x8 to 4x4
        self.fc = nn.Linear(128 * 4 * 4, 4096)  # 2048 inputs
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)  # 8x8 -> 4x4
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ChessModelV2(nn.Module):
    """
    Enhanced chess model with:
    1. Piece value embeddings (learns relative piece importance)
    2. Additional input channels for game state
    3. Residual connections for deeper learning
    4. Separate policy head for move prediction
    """
    
    def __init__(self, use_piece_values=True, num_residual_blocks=4):
        super().__init__()
        self.use_piece_values = use_piece_values
        
        # Input channels: 12 (pieces) + 7 (additional features) = 19
        # Additional features: castling rights (4), en passant (1), turn (1), piece values (1)
        input_channels = 19 if use_piece_values else 18
        
        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(128)
        
        # Residual blocks (inspired by AlphaZero)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(num_residual_blocks)
        ])
        
        # Policy head (move prediction)
        self.policy_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(64)
        self.policy_fc = nn.Linear(64 * 8 * 8, 4096)
        
        # Value head (position evaluation) - optional but helps learning
        self.value_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x, return_value=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, 8, 8)
            return_value: If True, also return position evaluation
            
        Returns:
            policy: Move logits (N, 4096)
            value: Position evaluation (N, 1) if return_value=True
        """
        # Initial convolution
        x = torch.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = torch.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        if return_value:
            # Value head
            value = torch.relu(self.value_bn(self.value_conv(x)))
            value = value.view(value.size(0), -1)
            value = torch.relu(self.value_fc1(value))
            value = torch.tanh(self.value_fc2(value))
            return policy, value
        
        return policy


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = torch.relu(x)
        return x


# Backward compatibility - use original model by default
ChessModel = ChessModelV2