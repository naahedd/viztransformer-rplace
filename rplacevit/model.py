import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import trainable_parameters

class SpatialEncoding2D(nn.Module):
    """
    [NO LONGER USED]
    2D positional encoding layer.
    We use almost the same positional encoding as the original transformer model,
    but we apply two different positional encodings for the width and height of the tensor, on different channels.
    
    Attributes:
        pos_enc (torch.Tensor): Positional encoding tensor.
    """
    def __init__(self, feat_dim: int, height: int, width: int, scale: float = 1.0):
        """
        Initialize the 2D positional encoding layer.
        
        Args:
            feat_dim (int): Number of features in the model.
            height (int): Height of the positional encoding tensor.
            width (int): Width of the positional encoding tensor.
            scale (float): Scaling factor for the positional encoding tensor.
            
        Raises:
            AssertionError: If feat_dim is not divisible by 4.
        """
        super().__init__()
        assert feat_dim % 4 == 0, "feat_dim must be divisible by 4"
        
        pos_enc = torch.zeros(feat_dim, height, width)
        feat_dim = int(feat_dim / 2)
        div_term = torch.exp(torch.arange(0., feat_dim, 2) * -(math.log(10000.0) / feat_dim))
        pos_w = torch.arange(0., width).unsqueeze(1) * scale
        pos_h = torch.arange(0., height).unsqueeze(1) * scale
        pos_enc[0:feat_dim:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pos_enc[1:feat_dim:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pos_enc[feat_dim::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pos_enc[feat_dim+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feat_dim, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, feat_dim, height, width).
        """
        return x + self.pos_enc
    
class OneHotSpatialEncoding2D(nn.Module):
    """
    [NO LONGER USED]
    2D one-hot positional encoding layer.
    We use one-hot positional encoding for the width and height of the tensor, on different channels.
    so 16 channels for width and 16 channels for height.
    
    Attributes:
        pos_enc (torch.Tensor): Positional encoding tensor.
    """
    def __init__(self, height, width):
        """
        Initialize the 2D one-hot positional encoding layer.
        
        Args:
            height (int): Height of the positional encoding tensor.
            width (int): Width of the positional encoding tensor.
        """
        super().__init__()
        
        pos_enc = torch.zeros(2, height, width)
        pos_w = torch.arange(0., width).long()
        pos_h = torch.arange(0., height).long()
        pos_enc[0, :, :] = F.one_hot(pos_w, num_classes=width)
        pos_enc[1, :, :] = F.one_hot(pos_h, num_classes=height)
        self.register_buffer('pos_enc', pos_enc)

class VaswaniSpatialEncoding(nn.Module):
    """
    Positional encoding layer from Vaswani et al. (2018).
    """
    def __init__(self, max_seq_len: int = 1024):
        """
        Initialize the positional encoding layer.
        
        Args:
            max_seq_len (int): Maximum length of the sequence.
        """
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, x):
        """
        Forward pass of the positional encoding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feat_dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, feat_dim).
        """
        if not hasattr(self, 'pos_enc') or self.pos_enc.size(0) < x.size(1) or self.pos_enc.size(1) != x.size(2):
            t = torch.arange(x.size(1), dtype=x.dtype, device=x.device)[:, None]
            j = torch.arange(x.size(2), dtype=x.dtype, device=x.device)[None, :]
            k = j % 2
            self.pos_enc = torch.sin(
                t / (self.max_seq_len ** ((j - k) / x.size(2))) + math.pi / 2 * k
            )

        return x + self.pos_enc[:x.size(1), :x.size(2)]

class VaswaniSpatialEncodingV2(nn.Module):
    """
    Positional encoding layer from Vaswani et al. (2018).
    """
    def __init__(self, feat_dim: int, max_seq_len: int = 10000):
        """
        Initialize the positional encoding layer.

        Args:
            feat_dim (int): Number of features in the model.
            max_seq_len (int): Maximum length of the sequence.
        """
        super().__init__()
        self.feat_dim = feat_dim

        pos_enc = torch.zeros(max_seq_len, feat_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feat_dim, 2).float() * (-math.log(10000.0) / feat_dim))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        """
        Forward pass of the positional encoding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feat_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, feat_dim).
        """
        seq_len = x.size(1)
        if seq_len > self.pos_enc.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.pos_enc.size(1)}")
        return x + self.pos_enc[:, :seq_len, :]

class MultiHeadAttentionLayer(nn.Module):
    """
    Multihead attention layer.
    El classico
    
    Attributes:
        q_proj (nn.Linear): Linear layer for query projection.
        k_proj (nn.Linear): Linear layer for key projection.
        v_proj (nn.Linear): Linear layer for value projection.
        out_proj (nn.Linear): Linear layer for output projection.
        dropout (nn.Dropout): Dropout layer
    """
    def __init__(self, feat_dim: int, num_heads: int, dropout: float):
        """
        Initialize the multihead attention layer.
        
        Args:
            feat_dim (int): Number of features in the model.
            num_heads (int): Number of heads in the multihead attention models.
            dropout (float): Dropout rate.
            
        Raises:
            AssertionError: If feat_dim is not divisible by num_heads.
        """
        super().__init__()
        assert feat_dim % num_heads == 0, "feat_dim must be divisible by num_heads"
        
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads

        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Forward pass of the multihead attention layer.
        Use scaled dot-product attention with query, key, and value tensors.
        
        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, feat_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, feat_dim).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, feat_dim).
            mask (torch.Tensor): Mask tensor of shape (batch_size, seq_len).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, feat_dim).
        """
        batch_size = q.size(0)
        
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.dropout else 0.0)
        
        context = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.feat_dim)
        
        return self.out_proj(context)
    
class MultiHeadAttentionLayerV2(nn.Module):
    """
    Multihead attention layer using PyTorch's built-in module.
    """
    def __init__(self, feat_dim: int, num_heads: int, dropout: float):
        """
        Initialize the multihead attention layer.

        Args:
            feat_dim (int): Number of features in the model.
            num_heads (int): Number of heads in the multihead attention models.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Forward pass of the multihead attention layer.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, feat_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, feat_dim).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, feat_dim).
            mask (torch.Tensor): Mask tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, feat_dim).
        """
        attn_output, _ = self.mha(q, k, v, key_padding_mask=mask)
        return self.dropout(attn_output)

class TransformerLayer(nn.Module):
    """
    Transformer block with multihead attention and feedforward network.
    
    Attributes:
        attn (MultiHeadAttentionLayer): Multihead attention layer.
        ff (nn.Sequential): Feedforward network.
        norm1 (nn.LayerNorm): Layer normalization layer.
        norm2 (nn.LayerNorm): Layer normalization layer.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, feat_dim: int, num_heads: int, ff_dim: int, dropout: float):
        """
        Initialize the transformer block.
        
        Args:
            feat_dim (int): Number of features in the model.
            num_heads (int): Number of heads in the multihead attention models.
            ff_dim (int): Number of features in the feedforward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        
        self.attn = MultiHeadAttentionLayer(feat_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(feat_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, feat_dim)
        )
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feat_dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, feat_dim).
        """
        normalized_x = self.norm1(x)
        x = x + self.dropout(self.attn(normalized_x, normalized_x, normalized_x))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
    
class TransformerLayerV2(nn.Module):
    """
    Transformer block with multihead attention and feedforward network.
    """
    def __init__(self, feat_dim: int, num_heads: int, ff_dim: int, dropout: float):
        """
        Initialize the transformer block.

        Args:
            feat_dim (int): Number of features in the model.
            num_heads (int): Number of heads in the multihead attention models.
            ff_dim (int): Number of features in the feedforward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        
        self.attn = MultiHeadAttentionLayer(feat_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(feat_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, feat_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feat_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, feat_dim).
        """
        # Multi-head attention sub-layer
        x_norm = self.norm1(x)
        attn_output = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        # Feedforward sub-layer
        x_norm = self.norm2(x)
        ff_output = self.ff(x_norm)
        x = x + ff_output

        return x
    
class PeripheralEncoderLayer(nn.Module):
    """
    [NO LONGER USED]
    Peripheral encoding layer.
    Encode the peripheral view of the model and scale it down 4 times.
    50% of the output will be the processed peripheral view and the other 50% will be the positional encoding.
    Might try to use other kernel sizes to encode each patch independently.
    
    Attributes:
        conv1 (nn.Conv2d): Convolutional layer for the first layer.
        conv2 (nn.Conv2d): Convolutional layer for the second layer.
        conv3 (nn.Conv2d): Convolutional layer for the third layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer for the first layer.
        bn2 (nn.BatchNorm2d): Batch normalization layer for the second layer.
        bn3 (nn.BatchNorm2d): Batch normalization layer for the third layer.
        pos_encoder (SpatialEncoding2D): Positional encoding
    """
    def __init__(self, in_channel=32, feat_dim=512, output_width=16, output_height=16):
        """
        Initialize the peripheral encoding layer.
        
        Args:
            in_channel (int): Number of input channels. Default is 32.
            feat_dim (int): Number of features in the model. Default is 512.
            output_width (int): Width of the output tensor. Default is 16.
            output_height (int): Height of the output tensor. Default is 16.
            
        Raises:
            AssertionError: If output_width is not divisible by 4 or feat_dim is not divisible by 8.
        """
        super().__init__()
        assert output_width % 4 == 0, "output_width must be divisible by 4"
        assert output_height % 4 == 0, "output_height must be divisible by 4"
        assert feat_dim % 8 == 0, "feat_dim must be divisible by 8"
        
        self.feat_dim = feat_dim
        self.output_width = output_width
        self.output_height = output_height
        self.conv1 = nn.Conv2d(in_channel + 1, feat_dim // 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feat_dim // 8)
        self.conv2 = nn.Conv2d(feat_dim // 8, feat_dim // 4, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(feat_dim // 4)
        self.conv3 = nn.Conv2d(feat_dim // 4, feat_dim // 2, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(feat_dim // 2)
        self.pos_encoder = SpatialEncoding2D(feat_dim // 2, output_width, output_height, scale=4.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the peripheral encoding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channel, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, feat_dim, output_width, output_height).
        """
        batch_size, in_channel, input_width, input_height = x.shape
        is_center = torch.zeros((batch_size, 1, input_width, input_height), device=x.device)
        x = torch.cat([x, is_center], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        pos_endoging = self.pos_encoder(torch.zeros((batch_size, self.feat_dim // 2, self.output_width, self.output_height), device=x.device))
        return torch.cat([x, pos_endoging], dim=1)
    
class CenterEncoderLayer(nn.Module):
    """
    [NO LONGER USED]
    Center encoding layer.
    Encode the center view of the model.
    50% of the output will be the processed center view and the other 50% will be the positional encoding.
    
    Attributes:
        conv1 (nn.Conv2d): Convolutional layer for the first layer.
        pos_encoder (SpatialEncoding2D): Positional encoding layer.
    """
    def __init__(self, in_channel=32, feat_dim=512, output_width=16, output_height=16):
        """
        Initialize the center encoding layer.
        
        Args:
            in_channel (int): Number of input channels. Default is 32.
            feat_dim (int): Number of features in the model. Default is 512.
            output_width (int): Width of the output tensor. Default is 16.
            output_height (int): Height of the output tensor. Default is 16.
            
        Raises:
            AssertionError: If feat_dim is not divisible by 2.
        """
        super().__init__()
        assert feat_dim % 2 == 0, "feat_dim must be divisible by 2"
        
        self.feat_dim = feat_dim
        self.output_width = output_width
        self.output_height = output_height
        self.conv1 = nn.Conv2d(in_channel + 1, feat_dim // 2, kernel_size=1)
        self.pos_encoder = SpatialEncoding2D(feat_dim // 2, output_width, output_height)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the center encoding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channel, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, feat_dim, output_width, output_height).
        """
        batch_size, in_channel, input_height, input_width = x.shape
        is_center = torch.ones((batch_size, 1, input_height, input_width), device=x.device)
        x = torch.cat([x, is_center], dim=1)
        x = F.relu(self.conv1(x))
        pos_endoging = self.pos_encoder(torch.zeros((batch_size, self.feat_dim // 2, self.output_width, self.output_height), device=x.device))
        return torch.cat([x, pos_endoging], dim=1)
    
class CenterEncoderLayerV2(nn.Module):
    """
    Center encoding layer.
    Encode the center view of the model pixel by pixel.
    
    Attributes:
        n_pixels (int): Number of pixels in the center view.
        proj (nn.Linear): Linear layer for projection.
        pos_encoder (VaswaniSpatialEncoding): Positional encoding
    """
    def __init__(self, width=16, feat_dim=512, in_channel=32):
        """
        Initialize the center encoding layer.
        
        Args:
            width (int): Width of the center view. Default is 16.
            feat_dim (int): Number of features in the model. Default is 512.
            in_channel (int): Number of input channels. Default is 32.
        """
        super().__init__()
        self.n_pixels = width * width
        
        self.proj = nn.Linear(in_channel, feat_dim)
        self.pos_encoder = VaswaniSpatialEncoding(len_max=self.n_pixels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the center encoding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channel, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_pixels, feat_dim).
        """
        batch_size, in_channel, height, width = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, self.n_pixels, -1)
        x = self.proj(x)
        x = self.pos_encoder(x)
        return x
    
class PeripheralEncoderLayerV2(nn.Module):
    """
    Peripheral encoding layer.
    Encode the peripheral view of the model and scale it down 4 times.
    From ViT paper.
    
    Attributes:
        n_patches (int): Number of patches.
        proj (nn.Linear): Linear layer for projection.
        pos_encoder (VaswaniSpatialEncoding): Positional encoding layer.
    """
    def __init__(self, width=64, patch_size=4, feat_dim=512, in_channel=32):
        """
        Initialize the peripheral encoding layer.
        
        Args:
            width (int): Width of the peripheral view. Default is 64.
            patch_size (int): Size of the patches. Default is 4.
            feat_dim (int): Number of features in the model. Default is 512.
            in_channel (int): Number of input channels. Default
            
        Raises:
            AssertionError: If width is not divisible by patch_size.
        """
        super().__init__()
        assert width % patch_size == 0, "width must be divisible by patch_size"
        
        self.width = width
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.n_patches = (width // patch_size) ** 2
        
        self.proj = nn.Linear(patch_size * patch_size * in_channel, feat_dim)
        self.pos_encoder = VaswaniSpatialEncoding(len_max=self.n_patches)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the peripheral encoding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channel, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_patches, feat_dim).
        """
        batch_size, in_channel, height, width = x.shape
        assert height == width == self.width, f"Input spatial dimensions should be {self.width}x{self.width}"
        assert in_channel == self.in_channel, f"Input should have {self.in_channel} channels"
        x = x.reshape(batch_size, in_channel, self.width // self.patch_size, self.patch_size, self.width // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(batch_size, self.n_patches, -1)
        x = self.proj(x)
        x = self.pos_encoder(x)
        return x
    
class ColorProjectionLayer(nn.Module):
    """
    Color projection layer.
    Project the feat_dim features to the number of output channels.
    
    Attributes:
        proj (nn.Linear): Linear layer for projection.
    """
    def __init__(self, feat_dim=512, out_channels=32):
        """
        Initialize the color projection layer.
        
        Args:
            feat_dim (int): Number of features in the model. Default is 512.
            out_channels (int): Number of output channels. Default is 32.
        """
        super().__init__()
        
        self.proj = nn.Linear(feat_dim, out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the color projection layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feat_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels).
        """
        return self.proj(x)
    
class TimeProjectionLayer(nn.Module):
    """
    [NO LONGER USED]
    Time projection layer.
    Project the feat_dim features to 1 channel.
    
    Attributes:
        proj (nn.Conv2d): Convolutional layer for projection.
    """
    def __init__(self, feat_dim=512):
        """
        Initialize the time projection layer.
        
        Args:
            feat_dim (int): Number of features in the model. Default is 512.
        """
        super().__init__()
        
        self.proj = nn.Conv2d(feat_dim, 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the time projection layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feat_dim, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, height, width).
        """
        return self.proj(x).squeeze(1)
    
class TimeProjectionLayerV2(nn.Module):
    """
    Time projection layer.
    Project the feat_dim features to 1 channel.
    
    Attributes:
        proj (nn.Linear): Linear layer for projection.
    """
    def __init__(self, feat_dim=512):
        """
        Initialize the time projection layer.
        
        Args:
            feat_dim (int): Number of features in the model. Default is 512.
        """
        super().__init__()
        
        self.proj = nn.Linear(feat_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the time projection layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feat_dim, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, height, width).
        """
        x = self.proj(x)
        return x.squeeze(-1).squeeze(-1)
    
class RPlaceColorTransformerModel(nn.Module):
    """
    [NO LONGER USED]
    A transformer-based model for predicting the next color of the center pixel in a view of the r/place canvas.
    This model uses a combination of center and peripheral encoders, followed by transformer blocks,
    to process the input and predict the next color for the center pixel.
    
    Attributes:
        center_encoder (CenterEncoderLayer): Center view encoder.
        peripheral_encoder (PeripheralEncoderLayer): Peripheral view encoder.
        transformer_blocks (nn.ModuleList): List of transformer blocks.
        final_norm (nn.LayerNorm): Final normalization layer.
        output_proj (ColorProjectionLayer): Output projection layer.
    """
    def __init__(self, in_channels=32, out_channels=32, feat_dim=512, num_heads=8, num_blocks=12, 
                 ff_dim=2048, dropout=0.0, center_width=16, peripheral_width=64, use_peripheral=False, output_strategy="cls_token"):
        """
        Initialize the RPlaceColorTransformer model.

        Args:
            in_channels (int): Number of input channels (32 colors from palette + 8 user classes). Default is 32.
            out_channels (int): Number of output channels (32 colors from palette). Default is 32.
            feat_dim (int): Number of features in the model. Default is 512.
            num_heads (int): Number of heads in the multihead attention models. Default is 8.
            num_layers (int): Number of transformer blocks. Default is 8.
            ff_dim (int): Number of features in the feedforward network. Default is 2048.
            dropout (float): Dropout rate. Default is 0.1.
            center_width (int): Width of the center view. Default is 16.
            peripheral_width (int): Width of the peripheral view. Default is 64.
            use_peripheral (bool): Whether to use the peripheral view. Default is False.
        """
        super().__init__()
        
        self.feat_dim = feat_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.center_width = center_width
        self.peripheral_width = peripheral_width
        self.use_peripheral = use_peripheral
        self.output_strategy = output_strategy
        
        self.center_encoder = CenterEncoderLayerV2(width=center_width, feat_dim=feat_dim, in_channel=in_channels)
        if use_peripheral:
            self.peripheral_encoder = PeripheralEncoderLayerV2(width=peripheral_width, feat_dim=feat_dim, in_channel=in_channels)
            
        if output_strategy == "cls_token":
            self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerLayer(feat_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(feat_dim)
        
        self.output_proj = ColorProjectionLayer(feat_dim, out_channels)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels).
        """
        batch_size, _, height, width = src.shape
        
        start_h = (height - self.center_width) // 2
        start_w = (width - self.center_width) // 2

        center_view = src[:, :, start_h:start_h+self.center_width, start_w:start_w+self.center_width]
        center_x = self.center_encoder(center_view)
        
        if self.use_peripheral:
            peripheral_view = src
            peripheral_x = self.peripheral_encoder(peripheral_view)
            x = torch.cat([center_x, peripheral_x], dim=1)
        else:
            x = center_x
            
        if self.output_strategy == "cls_token":
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)
        
        if self.output_strategy == "cls_token":
            cls_output = x[:, 0, :]
            output = self.output_proj(cls_output)
        elif self.output_strategy == "avg":
            avg_x = x.mean(dim=1)
            output = self.output_proj(avg_x)
        elif self.output_strategy == "center":
            center_pixel = x[:, self.center_width**2//2, :]
            output = self.output_proj(center_pixel)
        
        return output
    
    def init_weights(self):
        """
        Initialize the weights of the model, using Xavier uniform initialization.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
class RPlaceColorTransformerModelV2(nn.Module):
    """
    A transformer-based model for predicting the next color of the center pixel in a view of the r/place canvas.
    This model uses a combination of center and peripheral encoders, followed by transformer blocks,
    to process the input and predict the next color for the center pixel.
    
    Attributes:
        center_encoder (CenterEncoderLayer): Center view encoder.
        peripheral_encoder (PeripheralEncoderLayer): Peripheral view encoder.
        transformer_blocks (nn.ModuleList): List of transformer blocks.
        final_norm (nn.LayerNorm): Final normalization layer.
        output_proj (ColorProjectionLayer): Output projection layer.
    """
    def __init__(self, in_channels=32, out_channels=32, feat_dim=512, num_heads=8, num_blocks=12, 
                 ff_dim=2048, dropout=0.0, center_width=16, peripheral_width=64, use_peripheral=False, output_strategy="cls_token"):
        """
        Initialize the RPlaceColorTransformer model.

        Args:
            in_channels (int): Number of input channels (32 colors from palette + 8 user classes). Default is 32.
            out_channels (int): Number of output channels (32 colors from palette). Default is 32.
            feat_dim (int): Number of features in the model. Default is 512.
            num_heads (int): Number of heads in the multihead attention models. Default is 8.
            num_layers (int): Number of transformer blocks. Default is 8.
            ff_dim (int): Number of features in the feedforward network. Default is 2048.
            dropout (float): Dropout rate. Default is 0.1.
            center_width (int): Width of the center view. Default is 16.
            peripheral_width (int): Width of the peripheral view. Default is 64.
            use_peripheral (bool): Whether to use the peripheral view. Default is False.
            output_strategy (str): Output strategy. Default is "cls_token".
        """
        super().__init__()
        
        self.feat_dim = feat_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.center_width = center_width
        self.peripheral_width = peripheral_width
        self.use_peripheral = use_peripheral
        self.output_strategy = output_strategy

        self.center_encoder = CenterEncoderLayerV2(width=center_width, feat_dim=feat_dim, in_channel=in_channels)
        if use_peripheral:
            self.peripheral_encoder = PeripheralEncoderLayerV2(width=peripheral_width, feat_dim=feat_dim, in_channel=in_channels)
            
        if output_strategy == "cls_token":
            self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim))
            
        self.transformer_blocks = nn.ModuleList([
            TransformerLayerV2(feat_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(feat_dim)
        
        self.output_proj = ColorProjectionLayer(feat_dim, out_channels)
        
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels).
        """
        batch_size, _, height, width = src.shape
        
        start_h = (height - self.center_width) // 2
        start_w = (width - self.center_width) // 2

        center_view = src[:, :, start_h:start_h+self.center_width, start_w:start_w+self.center_width]
        center_x = self.center_encoder(center_view)
        
        if self.use_peripheral:
            peripheral_view = src
            peripheral_x = self.peripheral_encoder(peripheral_view)
            x = torch.cat([center_x, peripheral_x], dim=1)
        else:
            x = center_x
            
        if self.output_strategy == "cls_token":
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)
        
        if self.output_strategy == "cls_token":
            cls_output = x[:, 0, :]
            output = self.output_proj(cls_output)
        elif self.output_strategy == "avg":
            avg_x = x.mean(dim=1)
            output = self.output_proj(avg_x)
        elif self.output_strategy == "center":
            center_pixel = x[:, self.center_width**2//2, :]
            output = self.output_proj(center_pixel)
        
        return output
    
    def init_weights(self):
        """
        Initialize the weights of the model, using Xavier uniform initialization.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
class RPlaceTimeTransformerModel(nn.Module):
    """
    A transformer-based model for predicting the next pixel change in a view with a probability of changing the pixel.
    This model uses a combination of center and peripheral encoders, followed by transformer blocks,
    to process the input and predict the probability of changing the pixel for every pixel in the center view.
    
    Attributes:
        center_encoder (CenterEncoderLayer): Center view encoder.
        peripheral_encoder (PeripheralEncoderLayer): Peripheral view encoder.
        transformer_blocks (nn.ModuleList): List of transformer blocks.
        final_norm (nn.LayerNorm): Final normalization layer.
        output_proj (TimeProjectionLayerV2): Output projection layer.
    """
    def __init__(self, in_channels=32, feat_dim=512, num_heads=8, num_blocks=12, 
                 ff_dim=2048, dropout=0.0, center_width=16, peripheral_width=64, use_peripheral=False):
        """
        Initialize the RPlaceTimeTransformer model.
        
        Args:
            in_channels (int): Number of input channels (32 colors from palette + 8 user classes). Default is 32.
            feat_dim (int): Number of features in the model. Default is 512.
            num_heads (int): Number of heads in the multihead attention models. Default is 8.
            num_layers (int): Number of transformer blocks. Default is 10.
            ff_dim (int): Number of features in the feedforward network. Default is 2048.
            dropout (float): Dropout rate. Default is 0.1.
            center_width (int): Width of the center view. Default is 16.
            peripheral_width (int): Width of the peripheral view. Default is 64.
        """
        super().__init__()
        
        self.feat_dim = feat_dim
        self.in_channels = in_channels
        self.center_width = center_width
        self.peripheral_width = peripheral_width
        self.use_peripheral = use_peripheral

        self.center_encoder = CenterEncoderLayerV2(width=center_width, feat_dim=feat_dim, in_channel=in_channels)
        if use_peripheral:
            self.peripheral_encoder = PeripheralEncoderLayerV2(width=peripheral_width, feat_dim=feat_dim, in_channel=in_channels)

        self.transformer_blocks = nn.ModuleList([
            TransformerLayer(feat_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(feat_dim)

        self.output_proj = TimeProjectionLayerV2(feat_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, height, width).
        """
        batch_size, _, height, width = src.shape

        start_h = (height - self.center_width) // 2
        start_w = (width - self.center_width) // 2

        center_view = src[:, :, start_h:start_h+self.center_width, start_w:start_w+self.center_width]
        center_x = self.center_encoder(center_view)
        
        if self.use_peripheral:
            peripheral_view = src
            peripheral_x = self.peripheral_encoder(peripheral_view)
            x = torch.cat([center_x, peripheral_x], dim=1)
        else:
            x = center_x

        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)

        center_x = x[:, :self.center_width**2, :]
        center_x = center_x.view(batch_size, self.center_width, self.center_width, -1)
        
        output = self.output_proj(center_x)
        
        return output
    
    def init_weights(self):
        """
        Initialize the weights of the model, using Xavier uniform initialization.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
"""        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class RPlaceTimeCNN(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], in_channels=32):
        super(RPlaceTimeCNN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, 2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out.squeeze(-1)
"""

if __name__ == "__main__":
    input_tensor = torch.rand(4, 32, 64, 64)

    #model = RPlaceColorTransformerModel()
    #print(model)
    #print(f"Trainable parameters: {trainable_parameters(model)}")
    
    #output = model(input_tensor)
    #print(output.shape)
    
    #model = RPlaceTimeTransformerModelV2()
    #print(model)
    #print(f"Trainable parameters: {trainable_parameters(model)}")
    
    #output = model(input_tensor)
    #print(output.shape)
    #print(output)

    model = RPlaceColorTransformerModelV2()
    print(model)
    print(f"Trainable parameters: {trainable_parameters(model)}")