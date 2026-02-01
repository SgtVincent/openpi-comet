"""VLM2 Model with Pi-0.5 Integration.

This module implements the complete VLM2 model by integrating:
1. View-Consistent 3D-Aware Representation
2. Dual-Memory Module
3. Pi-0.5's transformer backbone and action decoder

The transformer and decoder parts use Pi-0.5's network structure (PaliGemma + Gemma Expert)
while the 3D-aware representation and memory modules are from VLM2.

Reference: 
- VLM2 paper: "Vision-Language Memory for Spatial Reasoning"
- Pi-0.5 paper: Physical Intelligence π0.5
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Literal, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.vlm2.view_consistent_3d import (
    ViewConsistent3DRepresentation,
    create_sinusoidal_3d_embedding,
)
from openpi.models_pytorch.vlm2.dual_memory import DualMemoryModule

# Type definitions for Gemma variants
GemmaVariant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]
PrecisionType = Literal["bfloat16", "float32"]

# Try to import Pi-0.5 components
try:
    from openpi.models_pytorch.pi0_pytorch import (
        PI0Pytorch,
        make_att_2d_masks,
        create_sinusoidal_pos_embedding,
        sample_beta,
    )
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
    import openpi.models.gemma as _gemma
    HAS_PI05 = True
except ImportError:
    HAS_PI05 = False
    _gemma = None  # type: ignore
    make_att_2d_masks = None  # type: ignore
    create_sinusoidal_pos_embedding = None  # type: ignore
    sample_beta = None  # type: ignore
    PaliGemmaWithExpertModel = None  # type: ignore
    logging.warning("Pi-0.5 components not found. VLM2WithPi05 will not be fully functional.")


@dataclass
class VLM2Config:
    """Configuration for VLM2 model.
    
    Combines VLM2-specific settings with Pi-0.5 settings.
    """
    # VLM2 specific settings
    visual_dim: int = 2048  # SigLIP output dimension
    geometry_dim: int = 512  # Geometry token dimension from 3D foundation model
    view_dim: int = 512  # View token dimension from 3D foundation model
    
    # Memory settings
    working_memory_size: int = 8  # Lw
    episodic_memory_capacity: int = 32  # Le
    episodic_similarity_threshold: float = 0.7  # τ
    episodic_fusion_alpha: float = 0.5  # α
    
    # Attention settings
    num_heads: int = 8
    hidden_dim: int = 1024
    dropout: float = 0.0
    
    # Pi-0.5 settings
    pi05: bool = True
    action_dim: int = 32
    action_horizon: int = 50
    dtype: PrecisionType = "bfloat16"
    paligemma_variant: GemmaVariant = "gemma_2b"
    action_expert_variant: GemmaVariant = "gemma_300m"
    
    # Video/frame settings
    num_frames: int = 32  # Number of frames to process
    frame_height: int = 224
    frame_width: int = 224
    patch_size: int = 16


from openpi.models_pytorch.vlm2.vggt_integration import VGGT3DEncoder


class VLM2PerceptionModule(nn.Module):
    """VLM2 Perception Module combining 3D encoding and representation.
    
    This module processes frames through:
    1. Optional 3D geometry encoder
    2. View-Consistent 3D-Aware Representation
    
    Args:
        config: VLM2 configuration
    """
    
    def __init__(self, config: VLM2Config):
        super().__init__()
        self.config = config
        
        # 3D Geometry Encoder (VGGT)
        self.geometry_encoder = VGGT3DEncoder(config)
        
        # View-Consistent 3D-Aware Representation
        self.view_consistent_3d = ViewConsistent3DRepresentation(
            visual_dim=config.visual_dim,
            geometry_dim=config.geometry_dim,
            view_dim=config.view_dim,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            pool_size=config.frame_height // config.patch_size,
            dropout=config.dropout,
        )
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Process visual tokens and images into 3D-aware representation.
        
        Args:
            visual_tokens: Visual tokens from vision encoder (batch, h, w, visual_dim)
            images: Source images (batch, 3, H, W) or (batch, C, H, W)
            
        Returns:
            3D-aware representation (batch, n, visual_dim)
        """
        # Encode geometry and view tokens from images using VGGT
        target_hw = (visual_tokens.shape[1], visual_tokens.shape[2])
        
        # Note: input visual_tokens is spatial (batch, h, w, c)
        # We need to ensure images are handled correctly. VGGT3DEncoder expects (B, S, C, H, W).
        # But here we might be processing frame-by-frame.
        # Let's check how it's called. It's called inside a loop over t.
        # So inputs here are (batch, h, w, dim) and (batch, C, H, W).
        # We need to unsqueeze time dimension for VGGT.
        
        images_seq = images.unsqueeze(1) # (B, 1, C, H, W)
        geometry_tokens, view_tokens, point_maps = self.geometry_encoder(images_seq, target_hw=target_hw)
        
        # Remove time dimension
        geometry_tokens = geometry_tokens.squeeze(1) # (B, h, w, dim)
        view_tokens = view_tokens.squeeze(1)
        point_maps = point_maps.squeeze(1)
        
        # Apply View-Consistent 3D-Aware Representation
        representation = self.view_consistent_3d(
            visual_tokens=visual_tokens,
            geometry_tokens=geometry_tokens,
            view_tokens=view_tokens,
            point_maps=point_maps,
        )
        
        return representation


class VLM2WithPi05(nn.Module):
    """VLM2 Model integrated with Pi-0.5 architecture.
    
    Combines:
    - VLM2's View-Consistent 3D-Aware Representation
    - VLM2's Dual-Memory Module  
    - Pi-0.5's PaliGemma (vision-language backbone)
    - Pi-0.5's Gemma Expert (action decoder)
    - Pi-0.5's Flow Matching for action generation
    
    The transformer and decoder parts use Pi-0.5's network structure,
    while perception and memory are from VLM2.
    
    Args:
        config: VLM2 configuration
    """
    
    def __init__(self, config: VLM2Config):
        super().__init__()
        self.config = config
        
        if not HAS_PI05:
            raise RuntimeError(
                "Pi-0.5 components are required but not available. "
                "Please ensure openpi.models_pytorch.pi0_pytorch is properly installed."
            )
        assert _gemma is not None
        assert PaliGemmaWithExpertModel is not None
        assert make_att_2d_masks is not None
        assert create_sinusoidal_pos_embedding is not None
        assert sample_beta is not None
        
        # Get Pi-0.5 model configurations
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        
        # Use actual visual_dim from PaliGemma config
        actual_visual_dim = paligemma_config.width
        
        # Update config with actual dimensions
        self.visual_dim = actual_visual_dim
        
        # Pi-0.5 backbone (PaliGemma + Gemma Expert)
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if config.pi05 else [False, False],
            precision=config.dtype,
        )
        
        # VLM2 Perception Module (View-Consistent 3D-Aware Representation)
        self.perception = VLM2PerceptionModule(
            VLM2Config(
                visual_dim=actual_visual_dim,
                geometry_dim=config.geometry_dim,
                view_dim=config.view_dim,
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                frame_height=config.frame_height,
                frame_width=config.frame_width,
                patch_size=config.patch_size,
            )
        )
        
        # VLM2 Dual-Memory Module
        self.memory = DualMemoryModule(
            feature_dim=actual_visual_dim,
            working_memory_size=config.working_memory_size,
            episodic_memory_capacity=config.episodic_memory_capacity,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            similarity_threshold=config.episodic_similarity_threshold,
            fusion_alpha=config.episodic_fusion_alpha,
        )
        
        # Pi-0.5 Action Projections (from pi0_pytorch.py)
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)
        
        # Pi-0.5 Time MLP for flow matching
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        
        # Store action config
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        
        # Projection to align 3D representation with language model dimension
        self.repr_to_llm = nn.Linear(actual_visual_dim, paligemma_config.width)
        
        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for VLM2WithPi05")
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for VLM2WithPi05")
    
    def process_video_with_memory(
        self,
        video_frames: torch.Tensor,
        text_query: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process video frames through perception and memory modules.
        
        Args:
            video_frames: Video frames (batch, num_frames, C, H, W)
            text_query: Optional text instruction embeddings (batch, len, dim)
            text_mask: Optional text mask (batch, len)
            
        Returns:
            Memory-enhanced representations (batch, num_frames, n_tokens, dim)
        """
        batch_size, num_frames, C, H, W = video_frames.shape
        device = video_frames.device
        
        # Reset memory for new sequence
        self.reset_memory(batch_size, device)
        
        all_representations = []
        
        for t in range(num_frames):
            # Get current frame
            frame = video_frames[:, t]  # (batch, C, H, W)
            
            # Get visual tokens from vision encoder
            visual_tokens = self.paligemma_with_expert.embed_image(frame)  # (batch, n, dim)
            
            # Reshape to spatial format
            n_tokens = visual_tokens.shape[1]
            h = w = int(math.sqrt(n_tokens))
            visual_tokens_spatial = rearrange(visual_tokens, 'b (h w) c -> b h w c', h=h, w=w)
            
            # Apply VLM2 perception (3D-aware representation)
            # Passes frame images to perception module which uses VGGT
            representation = self.perception(visual_tokens_spatial, frame)
            
            # Apply VLM2 memory with text query
            memory_enhanced = self.memory(
                representation, 
                text_query=text_query, 
                text_mask=text_mask,
                update_memory=True
            )
            
            all_representations.append(memory_enhanced)
        
        # Stack all representations
        representations = torch.stack(all_representations, dim=1)  # (batch, num_frames, n, dim)
        
        return representations
    
    def embed_action_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed noisy actions and timestep for expert processing.
        
        Args:
            noisy_actions: Noisy action sequence (batch, action_horizon, action_dim)
            timestep: Flow matching timestep (batch,)
            
        Returns:
            action_emb: Embedded actions
            pad_masks: Padding masks
            att_masks: Attention masks
            adarms_cond: AdaRMS conditioning (for Pi-0.5)
        """
        batch_size = noisy_actions.shape[0]
        device = noisy_actions.device
        
        # Embed timestep
        assert create_sinusoidal_pos_embedding is not None
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.to(dtype=timestep.dtype)
        
        # Embed actions
        action_emb = self.action_in_proj(noisy_actions)
        
        # Time MLP for AdaRMS
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = F.silu(time_emb)
        
        adarms_cond = time_emb
        
        # Create masks
        pad_masks = torch.ones(batch_size, self.action_horizon, dtype=torch.bool, device=device)
        att_masks = torch.tensor(
            [1] + [0] * (self.action_horizon - 1),
            dtype=action_emb.dtype,
            device=device,
        )
        att_masks = att_masks[None, :].expand(batch_size, -1)
        
        return action_emb, pad_masks, att_masks, adarms_cond
    
    def forward(
        self,
        video_frames: torch.Tensor,
        point_maps: torch.Tensor,
        language_tokens: torch.Tensor,
        language_masks: torch.Tensor,
        actions: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training.
        
        Args:
            video_frames: Video frames (batch, num_frames, C, H, W)
            point_maps: Point maps (batch, num_frames, H, W, 3) 
            language_tokens: Tokenized language instructions (batch, seq_len)
            language_masks: Language masks (batch, seq_len)
            actions: Ground truth actions (batch, action_horizon, action_dim)
            noise: Optional noise for flow matching
            time: Optional timestep for flow matching
            
        Returns:
            Loss tensor
        """
        # Note: point_maps argument is deprecated but kept for compatibility
        
        batch_size = actions.shape[0]
        device = actions.device
        
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(batch_size, device)
        assert time is not None
        
        # Embed language tokens FIRST to use for memory retrieval
        lang_emb = self.paligemma_with_expert.embed_language_tokens(language_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        
        # Flow matching interpolation
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        # Process video through VLM2 perception and memory
        # Pass language embeddings for query-guided retrieval
        memory_enhanced_repr = self.process_video_with_memory(
            video_frames, 
            text_query=lang_emb, 
            text_mask=language_masks
        )
        
        # Aggregate temporal representations (use last frame or mean)
        aggregated_repr = memory_enhanced_repr[:, -1]  # Use last frame: (batch, n, dim)
        
        # Project to language model dimension
        proj_dtype = self.repr_to_llm.weight.dtype
        prefix_embs = self.repr_to_llm(aggregated_repr.to(proj_dtype))
        
        # Language embeddings are already computed
        lang_emb = lang_emb.to(dtype=prefix_embs.dtype)
        prefix_embs = prefix_embs.to(lang_emb.dtype)
        
        # Combine visual and language embeddings
        prefix_embs = torch.cat([prefix_embs, lang_emb], dim=1)
        
        # Create prefix masks
        n_visual = memory_enhanced_repr.shape[2]
        prefix_pad_masks = torch.cat([
            torch.ones(batch_size, n_visual, dtype=torch.bool, device=device),
            language_masks,
        ], dim=1)
        prefix_att_masks = torch.zeros(prefix_embs.shape[1], dtype=torch.bool, device=device)
        prefix_att_masks = prefix_att_masks[None, :].expand(batch_size, -1)
        
        # Embed action suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_action_suffix(x_t, time)
        suffix_embs = suffix_embs.to(prefix_embs.dtype)
        
        # Combine prefix and suffix
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        
        # Create 2D attention masks
        assert make_att_2d_masks is not None
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        
        # Prepare 4D attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        
        # Forward through PaliGemma + Expert
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, adarms_cond],
        )
        
        if suffix_out is None:
            raise RuntimeError("Expected suffix outputs from PaliGemma expert forward pass.")
        
        # Extract relevant part of suffix output
        suffix_out = suffix_out[:, -self.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        
        # Project to action space
        v_t_pred = self.action_out_proj(suffix_out)
        
        # Calculate loss
        loss = F.mse_loss(v_t_pred, u_t, reduction="mean")
        
        return loss
    
    def reset_memory(self, batch_size: int, device: torch.device):
        """Reset memory for new sequences.
        
        Args:
            batch_size: Batch size
            device: Device
        """
        self.memory.reset(batch_size, device)
    
    def sample_noise(self, shape: Tuple, device: torch.device) -> torch.Tensor:
        """Sample noise for flow matching."""
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample time for flow matching."""
        assert sample_beta is not None
        time_beta = sample_beta(1.5, 1.0, batch_size, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)
    
    def _prepare_attention_masks_4d(self, att_2d_masks: torch.Tensor) -> torch.Tensor:
        """Prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train: bool = False):
        """Preprocess observation into frames and language tokens."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        images = list(observation.images.values())
        lang_tokens = observation.tokenized_prompt
        lang_masks = observation.tokenized_prompt_mask
        return images, lang_tokens, lang_masks

    def _build_video(
        self,
        images: List[torch.Tensor],
    ) -> torch.Tensor:
        """Build video frames from images."""
        if not images:
            raise ValueError("No images found in observation for VLM2 inference.")

        num_frames = self.config.num_frames
        if len(images) >= num_frames:
            frames = images[:num_frames]
        else:
            frames = images + [images[-1]] * (num_frames - len(images))

        video_frames = torch.stack(frames, dim=1)
        return video_frames
    
    
    @torch.no_grad()
    def sample_actions(
        self,
        device: torch.device,
        observation,
        noise: Optional[torch.Tensor] = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Sample actions using flow matching.

        Args:
            device: Torch device
            observation: Model observation
            noise: Optional initial noise
            num_steps: Number of denoising steps

        Returns:
            Sampled actions (batch, action_horizon, action_dim)
        """
        images, language_tokens, language_masks = self._preprocess_observation(observation, train=False)
        if language_tokens is None or language_masks is None:
            raise ValueError("Observation missing tokenized_prompt/tokenized_prompt_mask for VLM2 inference.")

        video_frames = self._build_video(images)
        batch_size = video_frames.shape[0]
        device = video_frames.device
        
        if noise is None:
            actions_shape = (batch_size, self.action_horizon, self.action_dim)
            noise = self.sample_noise(actions_shape, device)
            
        # Embed language tokens for memory retrieval
        lang_emb = self.paligemma_with_expert.embed_language_tokens(language_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        
        # Process video through VLM2 perception and memory
        memory_enhanced_repr = self.process_video_with_memory(
            video_frames,
            text_query=lang_emb,
            text_mask=language_masks
        )
        
        # Aggregate temporal representations
        aggregated_repr = memory_enhanced_repr[:, -1]
        
        # Project to language model dimension
        proj_dtype = self.repr_to_llm.weight.dtype
        prefix_embs = self.repr_to_llm(aggregated_repr.to(proj_dtype))
        
        # Reuse language embeddings
        lang_emb = lang_emb.to(dtype=prefix_embs.dtype)
        prefix_embs = prefix_embs.to(lang_emb.dtype)
        
        # Combine
        prefix_embs = torch.cat([prefix_embs, lang_emb], dim=1)
        
        # Create prefix masks
        n_visual = memory_enhanced_repr.shape[2]
        prefix_pad_masks = torch.cat([
            torch.ones(batch_size, n_visual, dtype=torch.bool, device=device),
            language_masks,
        ], dim=1)
        prefix_att_masks = torch.zeros(prefix_embs.shape[1], dtype=torch.bool, device=device)
        prefix_att_masks = prefix_att_masks[None, :].expand(batch_size, -1)
        
        # Create prefix attention masks
        assert make_att_2d_masks is not None
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        
        # Get KV cache from prefix
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        
        # Euler integration for sampling
        dt = -1.0 / num_steps
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        while time >= -dt / 2:
            expanded_time = time.expand(batch_size)
            v_t = self._denoise_step(
                x_t, expanded_time, prefix_pad_masks, past_key_values
            )
            x_t = x_t + dt * v_t
            time = time + dt
        
        return x_t
    
    def _denoise_step(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
    ) -> torch.Tensor:
        """Single denoising step.
        
        Args:
            x_t: Current noisy actions
            timestep: Current timestep
            prefix_pad_masks: Prefix padding masks
            past_key_values: KV cache from prefix
            
        Returns:
            Velocity prediction v_t
        """
        # Embed action suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_action_suffix(x_t, timestep)
        
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]
        
        # Create attention masks
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        assert make_att_2d_masks is not None
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        
        # Position IDs
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        
        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        
        # Forward through expert with KV cache
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        if suffix_out is None:
            raise RuntimeError("Expected suffix outputs from PaliGemma expert forward pass.")
        suffix_out = suffix_out[:, -self.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        
        return self.action_out_proj(suffix_out)
    
    def get_memory_stats(self) -> dict:
        """Get memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        return self.memory.get_memory_stats()


def create_vlm2_with_pi05(
    visual_dim: int = 2048,
    geometry_dim: int = 512,
    view_dim: int = 512,
    working_memory_size: int = 8,
    episodic_memory_capacity: int = 32,
    action_dim: int = 32,
    action_horizon: int = 50,
    **kwargs,
) -> VLM2WithPi05:
    """Factory function to create VLM2WithPi05 model.
    
    Args:
        visual_dim: Dimension of visual tokens
        geometry_dim: Dimension of geometry tokens
        view_dim: Dimension of view tokens
        working_memory_size: Size of working memory
        episodic_memory_capacity: Capacity of episodic memory
        action_dim: Action dimension
        action_horizon: Action horizon length
        **kwargs: Additional config parameters
        
    Returns:
        VLM2WithPi05 model instance
    """
    config = VLM2Config(
        visual_dim=visual_dim,
        geometry_dim=geometry_dim,
        view_dim=view_dim,
        working_memory_size=working_memory_size,
        episodic_memory_capacity=episodic_memory_capacity,
        action_dim=action_dim,
        action_horizon=action_horizon,
        **kwargs,
    )
    return VLM2WithPi05(config)
