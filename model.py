"""
Dual Attention Model architecture for RUL prediction.
Includes:
- Channel Attention mechanism
- Convolutional blocks with residual connections
- Sequence Attention mechanism
- GRU-based temporal modeling


Author: Aref Aasi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention module using both average and max pooling."""
    
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, in_channels)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, T)
        Returns:
            Attention-weighted tensor (B, C, T)
        """
        avg_out = self.fc2(torch.tanh(self.fc1(self.avg_pool(x).squeeze(-1))))
        max_out = self.fc2(torch.tanh(self.fc1(self.max_pool(x).squeeze(-1))))
        attn = F.softmax(avg_out + max_out, dim=1).unsqueeze(-1)
        return x * attn


class ConvBlock(nn.Module):
    """Convolutional block with channel attention and residual connection."""
    
    def __init__(self, in_ch, out_ch, k=3, dilation=1, p_drop=0.2):
        super().__init__()
        pad = (k - 1)//2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.SiLU()
        self.ca   = ChannelAttention(out_ch)
        self.drop = nn.Dropout(p_drop)
        self.res  = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C_in, T)
        Returns:
            Output tensor (B, C_out, T)
        """
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        y = self.ca(y)
        y = self.drop(y)
        return y + self.res(x)


class SequenceAttention(nn.Module):
    """Sequence attention mechanism for temporal modeling."""
    
    def __init__(self, hidden):
        super().__init__()
        self.attn_fc = nn.Linear(hidden, 1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, T, H)
        Returns:
            Attention-weighted output (B, H)
        """
        scores = F.softmax(self.attn_fc(x).squeeze(-1), dim=1)  # (B, T)
        return torch.bmm(scores.unsqueeze(1), x).squeeze(1)     # (B, H)


class DualAttentionModel(nn.Module):
    """
    Dual Attention Model combining channel and sequence attention.
    
    Architecture:
        1. Multi-scale CNN with channel attention
        2. Bidirectional GRU for temporal modeling
        3. Sequence attention for aggregation
        4. Fully connected layers for RUL prediction
    """
    
    def __init__(
        self,
        num_features,
        seq_len=80,
        hidden_gru=256,
        p_drop=0.2,
        conv_channels=None,
        kernel_sizes=None,
        dilations=None
    ):
        """
        Args:
            num_features: Number of input features
            seq_len: Sequence length (not directly used in forward)
            hidden_gru: Hidden size for GRU
            p_drop: Dropout probability
            conv_channels: List of channel sizes for conv layers
            kernel_sizes: List of kernel sizes for conv layers
            dilations: List of dilation rates for conv layers
        """
        super().__init__()
        
        if conv_channels is None:
            conv_channels = [16, 16, 32, 32, 64]
        if kernel_sizes is None:
            kernel_sizes  = [5, 3, 3, 3, 3]
        if dilations is None:
            dilations     = [1, 1, 2, 4, 1]

        # Build convolutional layers
        layers, in_ch = [], num_features
        for out_ch, k, d in zip(conv_channels, kernel_sizes, dilations):
            layers.append(ConvBlock(in_ch, out_ch, k=k, dilation=d, p_drop=p_drop))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # Recurrent and attention layers
        self.gru = nn.GRU(input_size=in_ch, hidden_size=hidden_gru, batch_first=True)
        self.sam = SequenceAttention(hidden_gru)
        
        # Prediction head
        self.fc  = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(hidden_gru, 64),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, T, F)
        Returns:
            Predicted RUL values (B,)
        """
        x = x.permute(0, 2, 1)      # (B, F, T)
        x = self.conv(x)            # (B, C, T)
        x = x.permute(0, 2, 1)      # (B, T, C)
        out, _ = self.gru(x)        # (B, T, H)
        h = self.sam(out)           # (B, H)
        y = self.fc(h).squeeze(-1)  # (B,)
        return y
