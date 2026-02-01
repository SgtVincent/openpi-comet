"""View-Consistent 3D-Aware Representation module.

This module implements the 3D-aware representation components from VLM2:
1. Adaptive 3D Position Injection
2. Viewpoint-Aware Geometry Alignment  
3. Semantic-Geometric Fusion

Reference: VLM2 paper Section 3.2
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
    return target_dtype


def create_sinusoidal_3d_embedding(
    coords: torch.Tensor,
    embedding_dim: int,
    min_period: float = 0.01,
    max_period: float = 10.0,
) -> torch.Tensor:
    """Create sinusoidal positional embeddings for 3D coordinates.
    
    Args:
        coords: 3D coordinates tensor of shape (batch, h, w, 3) or (batch, n, 3)
        embedding_dim: Output embedding dimension (must be divisible by 6 for x,y,z each)
        min_period: Minimum period for sinusoidal encoding
        max_period: Maximum period for sinusoidal encoding
        
    Returns:
        Positional embeddings of shape (batch, h, w, embedding_dim) or (batch, n, embedding_dim)
    """
    if embedding_dim % 6 != 0:
        # Adjust to make it work with any dimension by distributing across x,y,z
        dim_per_axis = embedding_dim // 3
        if dim_per_axis % 2 != 0:
            dim_per_axis = (dim_per_axis // 2) * 2
    else:
        dim_per_axis = embedding_dim // 3
    
    device = coords.device
    dtype = get_safe_dtype(torch.float32, device.type)
    
    # Create frequency bands
    fraction = torch.linspace(0.0, 1.0, dim_per_axis // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    
    embeddings = []
    for i in range(3):  # x, y, z
        coord = coords[..., i:i+1]  # (..., 1)
        sin_input = scaling_factor.view(1, 1, -1) * coord  # (..., dim_per_axis//2)
        if coord.dim() == 4:  # (batch, h, w, 1)
            sin_input = sin_input.squeeze(-2)  # (batch, h, w, dim_per_axis//2)
        emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)
        embeddings.append(emb)
    
    # Concatenate all axis embeddings
    pos_emb = torch.cat(embeddings, dim=-1)
    
    # Pad or truncate to exact embedding_dim if needed
    if pos_emb.shape[-1] < embedding_dim:
        padding = torch.zeros(*pos_emb.shape[:-1], embedding_dim - pos_emb.shape[-1], 
                            device=device, dtype=pos_emb.dtype)
        pos_emb = torch.cat([pos_emb, padding], dim=-1)
    elif pos_emb.shape[-1] > embedding_dim:
        pos_emb = pos_emb[..., :embedding_dim]
    
    return pos_emb


class Adaptive3DPositionInjection(nn.Module):
    """Adaptive 3D Position Injection module.
    
    Injects predicted 3D coordinates into visual tokens while handling noisy predictions
    through a learnable gating mechanism.
    
    Reference: VLM2 paper Section 3.2 - Adaptive 3D Position Injection
    
    Args:
        visual_dim: Dimension of visual tokens (c)
        hidden_dim: Hidden dimension for the position encoding MLP
        pool_size: Kernel size for pooling point maps to patch resolution
    """
    
    def __init__(
        self,
        visual_dim: int,
        hidden_dim: Optional[int] = None,
        pool_size: int = 14,  # Default for 224x224 images with 16x16 patches -> 14x14 patches
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim or visual_dim
        self.pool_size = pool_size
        
        # 3D coordinate to embedding MLP: φ: R^3 → R^c
        # Two-layer MLP as described in the paper
        self.pos_mlp = nn.Sequential(
            nn.Linear(visual_dim, self.hidden_dim),  # Input is from sinusoidal encoding
            nn.GELU(),
            nn.Linear(self.hidden_dim, visual_dim),
        )
        
        # Learnable gate αt ∈ [0,1]^(h×w×1)
        # Implemented as a small network that outputs gate values
        self.gate_network = nn.Sequential(
            nn.Linear(visual_dim * 2, self.hidden_dim),  # concat(visual, pos_emb)
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Adaptive pooling for point maps
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
    
    def pool_point_maps(self, point_maps: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Pool point maps from (H, W, 3) to (h, w, 3).
        
        Args:
            point_maps: Point maps of shape (batch, H, W, 3)
            target_h: Target height (patch height)
            target_w: Target width (patch width)
            
        Returns:
            Pooled coordinates of shape (batch, h, w, 3)
        """
        batch_size = point_maps.shape[0]
        # Reshape to (batch, 3, H, W) for pooling
        point_maps = rearrange(point_maps, 'b h w c -> b c h w')
        
        # Use adaptive pooling to get target resolution
        pooled = F.adaptive_avg_pool2d(point_maps, (target_h, target_w))
        
        # Reshape back to (batch, h, w, 3)
        pooled = rearrange(pooled, 'b c h w -> b h w c')
        
        return pooled
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        point_maps: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive 3D position injection.
        
        Args:
            visual_tokens: Visual feature tokens of shape (batch, h, w, c) or (batch, n, c)
            point_maps: Per-pixel 3D coordinates of shape (batch, H, W, 3)
            
        Returns:
            Position-aware visual tokens of shape (batch, h, w, c) or (batch, n, c)
        """
        original_shape = visual_tokens.shape
        is_sequence = visual_tokens.dim() == 3
        
        if is_sequence:
            # Assume square spatial layout for sequence input
            batch_size, n, c = visual_tokens.shape
            h = w = int(math.sqrt(n))
            visual_tokens = rearrange(visual_tokens, 'b (h w) c -> b h w c', h=h, w=w)
        else:
            batch_size, h, w, c = visual_tokens.shape
        
        # Pool point maps to patch resolution: Ct ∈ R^(h×w×3)
        coords = self.pool_point_maps(point_maps, h, w)

        # Encode 3D coordinates using sinusoidal encoding
        pos_emb = create_sinusoidal_3d_embedding(coords, self.visual_dim)

        # Align dtypes for MLP and gate networks
        original_dtype = visual_tokens.dtype
        target_dtype = self.pos_mlp[0].weight.dtype
        visual_tokens = visual_tokens.to(target_dtype)
        pos_emb = pos_emb.to(target_dtype)

        # Apply position MLP: φ(Ct)
        pos_features = self.pos_mlp(pos_emb)

        # Compute adaptive gate: αt = σ(gate_network(concat(visual, pos)))
        gate_input = torch.cat([visual_tokens, pos_features], dim=-1)
        gate = self.gate_network(gate_input)  # (batch, h, w, 1)

        # Apply gated injection: F_pa_t = Ft + αt ⊙ φ(Ct)
        position_aware_tokens = visual_tokens + gate * pos_features
        position_aware_tokens = position_aware_tokens.to(original_dtype)
        
        if is_sequence:
            position_aware_tokens = rearrange(position_aware_tokens, 'b h w c -> b (h w) c')
        
        return position_aware_tokens


class ViewpointAwareGeometryAlignment(nn.Module):
    """Viewpoint-Aware Geometry Alignment module.
    
    Aligns geometry tokens with view tokens to resolve viewpoint ambiguity,
    making geometric features viewpoint-aware.
    
    Reference: VLM2 paper Section 3.2 - Viewpoint-Aware Geometry Alignment
    
    Args:
        geometry_dim: Dimension of geometry tokens (cg)
        view_dim: Dimension of view tokens (cv)
        hidden_dim: Hidden dimension for fusion MLP
    """
    
    def __init__(
        self,
        geometry_dim: int,
        view_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.geometry_dim = geometry_dim
        self.view_dim = view_dim
        self.hidden_dim = hidden_dim or geometry_dim
        
        # Projection for patch-level view tokens: ψv: R^cv → R^cg
        self.view_proj_patch = nn.Linear(view_dim, geometry_dim)
        
        # Projection for global view descriptor: ψg: R^cv → R^cg  
        self.view_proj_global = nn.Linear(view_dim, geometry_dim)
        
        # Fusion MLP for combining geometry and view tokens
        self.fusion_mlp = nn.Sequential(
            nn.Linear(geometry_dim * 2, self.hidden_dim),  # concat(G, Ẑ)
            nn.GELU(),
            nn.Linear(self.hidden_dim, geometry_dim),
        )
    
    def forward(
        self,
        geometry_tokens: torch.Tensor,
        view_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Apply viewpoint-aware geometry alignment.
        
        Args:
            geometry_tokens: Geometry tokens of shape (batch, h, w, cg) or (batch, n, cg)
            view_tokens: View tokens of shape (batch, h, w, cv) or (batch, n, cv)
            
        Returns:
            Viewpoint-consistent geometry tokens of shape (batch, hw+1, cg)
        """
        is_sequence = geometry_tokens.dim() == 3
        
        if is_sequence:
            batch_size, n, cg = geometry_tokens.shape
            h = w = int(math.sqrt(n))
            geometry_tokens = rearrange(geometry_tokens, 'b (h w) c -> b h w c', h=h, w=w)
            view_tokens = rearrange(view_tokens, 'b (h w) c -> b h w c', h=h, w=w)
        else:
            batch_size, h, w, cg = geometry_tokens.shape
        
        # Project view tokens to geometry dimension: Ẑt = ψv(Zt)
        projected_view = self.view_proj_patch(view_tokens)  # (batch, h, w, cg)
        
        # Fuse geometry with projected view tokens: G_va_t = MLP(Concat[Gt; Ẑt])
        fused_input = torch.cat([geometry_tokens, projected_view], dim=-1)
        viewpoint_aware = self.fusion_mlp(fused_input)  # (batch, h, w, cg)
        
        # Compute global view descriptor: Z̄t = pool(Zt)
        global_view = view_tokens.mean(dim=(1, 2), keepdim=False)  # (batch, cv)
        
        # Project global descriptor: Z_g_t = ψg(Z̄t)
        global_view_proj = self.view_proj_global(global_view)  # (batch, cg)
        global_view_proj = global_view_proj.unsqueeze(1)  # (batch, 1, cg)
        
        # Flatten spatial dimensions
        viewpoint_aware = rearrange(viewpoint_aware, 'b h w c -> b (h w) c')
        
        # Concatenate with global descriptor: G_vc_t = Concat[G_va_t; Z_g_t]
        viewpoint_consistent = torch.cat([viewpoint_aware, global_view_proj], dim=1)
        # Shape: (batch, hw+1, cg)
        
        return viewpoint_consistent


class SemanticGeometricFusion(nn.Module):
    """Semantic-Geometric Fusion module using Cross-Attention.
    
    Fuses position-aware visual tokens with viewpoint-aware geometry tokens
    using cross-attention to produce the final 3D-aware representation.
    
    Reference: VLM2 paper Section 3.2 - Semantic-Geometric Fusion
    
    Args:
        visual_dim: Dimension of visual tokens (c)
        geometry_dim: Dimension of geometry tokens (cg)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        visual_dim: int,
        geometry_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.geometry_dim = geometry_dim
        self.num_heads = num_heads
        
        # Ensure dimensions are divisible by num_heads
        assert visual_dim % num_heads == 0, \
            f"visual_dim ({visual_dim}) must be divisible by num_heads ({num_heads})"
        
        self.head_dim = visual_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection from visual tokens
        self.q_proj = nn.Linear(visual_dim, visual_dim)
        
        # Key and Value projections from geometry tokens
        self.k_proj = nn.Linear(geometry_dim, visual_dim)
        self.v_proj = nn.Linear(geometry_dim, visual_dim)
        
        # Output projection
        self.out_proj = nn.Linear(visual_dim, visual_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        geometry_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply cross-attention fusion.
        
        Args:
            visual_tokens: Position-aware visual tokens (batch, n_visual, visual_dim)
            geometry_tokens: Viewpoint-aware geometry tokens (batch, n_geo, geometry_dim)
            attention_mask: Optional attention mask (batch, n_visual, n_geo)
            
        Returns:
            3D-aware tokens of shape (batch, n_visual, visual_dim)
        """
        batch_size, n_visual, _ = visual_tokens.shape
        _, n_geo, _ = geometry_tokens.shape

        # Align dtypes for projections
        original_dtype = visual_tokens.dtype
        target_dtype = self.q_proj.weight.dtype
        visual_tokens = visual_tokens.to(target_dtype)
        geometry_tokens = geometry_tokens.to(target_dtype)
        
        # Compute Q, K, V
        q = self.q_proj(visual_tokens)  # (batch, n_visual, visual_dim)
        k = self.k_proj(geometry_tokens)  # (batch, n_geo, visual_dim)
        v = self.v_proj(geometry_tokens)  # (batch, n_geo, visual_dim)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, n_visual, n_geo)
        
        if attention_mask is not None:
            attn = attn.masked_fill(~attention_mask.unsqueeze(1), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, heads, n_visual, head_dim)
        
        # Reshape back
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Output projection
        out = self.out_proj(out)
        out = out.to(original_dtype)
        
        return out


class ViewConsistent3DRepresentation(nn.Module):
    """Complete View-Consistent 3D-Aware Representation module.
    
    Combines all three components:
    1. Adaptive 3D Position Injection
    2. Viewpoint-Aware Geometry Alignment
    3. Semantic-Geometric Fusion
    
    Reference: VLM2 paper Section 3.2
    
    Args:
        visual_dim: Dimension of visual tokens from vision encoder
        geometry_dim: Dimension of geometry tokens from 3D foundation model
        view_dim: Dimension of view tokens from 3D foundation model
        num_heads: Number of attention heads for fusion
        hidden_dim: Hidden dimension for MLPs (default: same as visual_dim)
        pool_size: Kernel size for pooling point maps (default: 14 for 224x224 input)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        visual_dim: int,
        geometry_dim: int,
        view_dim: int,
        num_heads: int = 8,
        hidden_dim: Optional[int] = None,
        pool_size: int = 14,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.geometry_dim = geometry_dim
        self.view_dim = view_dim
        
        # 1. Adaptive 3D Position Injection
        self.position_injection = Adaptive3DPositionInjection(
            visual_dim=visual_dim,
            hidden_dim=hidden_dim,
            pool_size=pool_size,
        )
        
        # 2. Viewpoint-Aware Geometry Alignment
        self.geometry_alignment = ViewpointAwareGeometryAlignment(
            geometry_dim=geometry_dim,
            view_dim=view_dim,
            hidden_dim=hidden_dim,
        )
        
        # 3. Semantic-Geometric Fusion
        self.fusion = SemanticGeometricFusion(
            visual_dim=visual_dim,
            geometry_dim=geometry_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Residual connection with layer norm
        self.layer_norm = nn.LayerNorm(visual_dim)
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        geometry_tokens: torch.Tensor,
        view_tokens: torch.Tensor,
        point_maps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute view-consistent 3D-aware representation.
        
        Args:
            visual_tokens: Visual tokens from vision encoder (batch, h, w, c) or (batch, n, c)
            geometry_tokens: Geometry tokens from 3D foundation model (batch, h, w, cg) or (batch, n, cg)
            view_tokens: View tokens from 3D foundation model (batch, h, w, cv) or (batch, n, cv)
            point_maps: Per-pixel 3D coordinates (batch, H, W, 3)
            attention_mask: Optional attention mask for fusion
            
        Returns:
            3D-aware representation tokens of shape (batch, n, visual_dim)
        """
        is_spatial = visual_tokens.dim() == 4
        
        # 1. Adaptive 3D Position Injection: F_pa_t = Ft + αt ⊙ φ(Ct)
        position_aware_tokens = self.position_injection(visual_tokens, point_maps)
        
        # Flatten spatial dimensions if needed
        if is_spatial:
            position_aware_tokens = rearrange(position_aware_tokens, 'b h w c -> b (h w) c')
        
        # 2. Viewpoint-Aware Geometry Alignment: G_vc_t
        viewpoint_aware_geometry = self.geometry_alignment(geometry_tokens, view_tokens)
        
        # 3. Semantic-Geometric Fusion: Ht = CrossAttn(F_pa_t, G_vc_t)
        fused_representation = self.fusion(
            position_aware_tokens,
            viewpoint_aware_geometry,
            attention_mask,
        )
        
        # Residual connection and normalization
        original_dtype = position_aware_tokens.dtype
        target_dtype = self.layer_norm.weight.dtype
        position_aware_tokens = position_aware_tokens.to(target_dtype)
        fused_representation = fused_representation.to(target_dtype)
        output = self.layer_norm(position_aware_tokens + fused_representation)
        output = output.to(original_dtype)
        
        return output
