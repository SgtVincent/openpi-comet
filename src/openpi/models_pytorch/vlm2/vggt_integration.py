import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Helper to add path
def _ensure_vggt_path():
    current_file = os.path.abspath(__file__)
    # .../src/openpi/models_pytorch/vlm2/vggt_integration.py -> .../src/openpi/third_party/vggt
    src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file)))))
    vggt_path = os.path.join(src_root, "src", "openpi", "third_party", "vggt")
    if vggt_path not in sys.path:
        sys.path.append(vggt_path)

_ensure_vggt_path()

try:
    from vggt.models.vggt import VGGT
except ImportError:
    VGGT = None

class VGGT3DEncoder(nn.Module):
    """
    Wrapper around VGGT to serve as the Geometry3DEncoder for VLM2.
    
    Extracts:
    1. point_maps (world_points)
    2. geometry_tokens (visual features from aggregator)
    3. view_tokens (camera pose embeddings)
    """
    def __init__(self, config):
        super().__init__()
        if VGGT is None:
            raise ImportError("VGGT module not found. Please clone facebookresearch/vggt into src/openpi/third_party/vggt")
            
        # Initialize VGGT
        # Note: We rely on default parameters or minimal config here.
        # In a real deployment, we should load pre-trained weights.
        # For now, we instantiate the model structure.
        self.model = VGGT(
            img_size=config.frame_height, 
            patch_size=config.patch_size,
            embed_dim=1024, # Standard for VGGT
            enable_camera=True,
            enable_point=True,
            enable_depth=True, # We might use depth later
            enable_track=False # Track head not needed for basic VLM2 perception
        )
        
        # Dimensions
        self.geometry_dim = config.geometry_dim
        self.view_dim = config.view_dim
        vggt_embed_dim = 2048 # Aggregator returns concatenated frame+global tokens (2*1024)
        
        # Projections
        self.geometry_proj = nn.Linear(vggt_embed_dim, config.geometry_dim)
        
        # Camera head output is dim 9 (absT_quaR_FoV)
        camera_dim = 9 
        self.view_proj = nn.Linear(camera_dim, config.view_dim)
        
        self.patch_size = config.patch_size

    def forward(
        self, 
        images: torch.Tensor,
        target_hw: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, S, C, H, W) normalized or not? VGGT handles normalization.
            target_hw: target height/width for interpolation (optional)
            
        Returns:
            geometry_tokens: (B, S, h, w, geometry_dim)
            view_tokens: (B, S, h, w, view_dim)
            point_maps: (B, S, H, W, 3)
        """
        B, S, C, H, W = images.shape
        
        # 1. Forward through Aggregator
        # Aggregator expects (B, S, C, H, W)
        aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
        features = aggregated_tokens_list[-1] 
        # features shape is (B, S, P, D) from Aggregator
        
        # Reshape to (B*S, P, D)
        features = features.reshape(B*S, -1, features.shape[-1])
        
        # 2. Extract Point Maps (World Points)
        if self.model.point_head is not None:
             point_maps, _ = self.model.point_head(
                 aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
             ) 
             # point_maps shape: (B, S, H, W, 3) usually?
             # VGGT doc says: world_points (torch.Tensor): [B, S, H, W, 3]
        else:
             # Fallback if disabled (shouldn't happen)
             point_maps = torch.zeros(B, S, H, W, 3, device=images.device)
             
        # 3. Extract Geometry Tokens
        # features: (B*S, P, D). P = special_tokens + patch_tokens
        # We need to slice off the special tokens.
        patch_tokens = features[:, patch_start_idx:, :] # (B*S, N_patches, D)
        
        # Project D -> geometry_dim
        geometry_tokens = self.geometry_proj(patch_tokens)
        
        # Reshape to spatial
        # N_patches = (H // patch_size) * (W // patch_size)
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        
        geometry_tokens = geometry_tokens.view(B, S, h_patches, w_patches, self.geometry_dim)
        
        # 4. Extract View Tokens
        if self.model.camera_head is not None:
            pose_enc_list = self.model.camera_head(aggregated_tokens_list)
            pose_enc = pose_enc_list[-1] # (B, S, 9)
        else:
            pose_enc = torch.zeros(B, S, 9, device=images.device)
            
        # Project pose 9 -> view_dim
        view_emb = self.view_proj(pose_enc) # (B, S, view_dim)
        
        # Broadcast to spatial: (B, S, h_patches, w_patches, view_dim)
        view_tokens = view_emb.unsqueeze(2).unsqueeze(3).expand(-1, -1, h_patches, w_patches, -1)
        
        # Interpolate if target_hw provided and different
        if target_hw is not None:
            th, tw = target_hw
            if (h_patches, w_patches) != (th, tw):
                # Reshape for interpolate: (B*S, dim, h, w)
                geometry_tokens = geometry_tokens.view(B*S, h_patches, w_patches, self.geometry_dim).permute(0, 3, 1, 2)
                view_tokens = view_tokens.view(B*S, h_patches, w_patches, self.view_dim).permute(0, 3, 1, 2)
                
                geometry_tokens = F.interpolate(geometry_tokens, size=(th, tw), mode='bilinear', align_corners=False)
                view_tokens = F.interpolate(view_tokens, size=(th, tw), mode='bilinear', align_corners=False)
                
                # Reshape back
                geometry_tokens = geometry_tokens.permute(0, 2, 3, 1).view(B, S, th, tw, self.geometry_dim)
                view_tokens = view_tokens.permute(0, 2, 3, 1).view(B, S, th, tw, self.view_dim)
        
        # Note: point_maps is (B, S, H, W, 3). Position injection handles pooling internally,
        # so we perform full resolution point_map return.
        
        return geometry_tokens, view_tokens, point_maps
