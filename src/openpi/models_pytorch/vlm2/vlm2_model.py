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

import dataclasses
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Literal, TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from openpi.models_pytorch.dtype_utils import align_tensors_to_reference_dtype
from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.vlm2.view_consistent_3d import (
    ViewConsistent3DRepresentation,
    create_sinusoidal_3d_embedding,
)
from openpi.models_pytorch.vlm2.dual_memory import DualMemoryModule

# Type definitions for Gemma variants
GemmaVariant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]
PrecisionType = Literal["bfloat16", "float16", "float32"]

# Try to import Pi-0.5 components
try:
    from openpi.models_pytorch.pi0_pytorch import (
        PI0Pytorch,
        make_att_2d_masks,
        create_sinusoidal_pos_embedding,
        sample_beta,
    )
    from openpi.models_pytorch.cache_utils import PreserveCacheLen
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
    import openpi.models.gemma as _gemma
    import openpi.shared.download as _download
    import sentencepiece as _sentencepiece
    HAS_PI05 = True
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_PI05 = False
    HAS_SENTENCEPIECE = False
    _gemma = None  # type: ignore
    _download = None  # type: ignore
    _sentencepiece = None  # type: ignore
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

    sem_geo_fusion_tanh_gate_enable: bool = False
    sem_geo_fusion_tanh_gate_init_alpha: float = 0.0
    
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

    vggt_pretrained: str | None = None
    vggt_load_strict: bool = False
    vggt_enable_track: bool = False
    freeze_vggt_backbone: bool = False
    freeze_image_encoder: bool = False


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
    
    def __init__(
        self,
        config: VLM2Config,
        *,
        action_expert_name: str = "gemma_token",
        action_expert_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.config = config
        if action_expert_name != "gemma_token":
            raise ValueError(f"VLM2WithPi05 currently supports only action_expert_name='gemma_token', got {action_expert_name}")
        
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
        
        t0 = time.perf_counter()
        logging.info("VLM2WithPi05 init: creating PaliGemmaWithExpertModel (paligemma=%s expert=%s)",
                     config.paligemma_variant, config.action_expert_variant)
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if config.pi05 else [False, False],
            precision=config.dtype,
        )
        logging.info("VLM2WithPi05 init: PaliGemmaWithExpertModel created in %.2fs", time.perf_counter() - t0)

        if config.freeze_image_encoder:
            for p in self.paligemma_with_expert.paligemma.vision_tower.parameters():
                p.requires_grad = False
        
        t1 = time.perf_counter()
        logging.info("VLM2WithPi05 init: creating VLM2PerceptionModule")
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
                vggt_pretrained=config.vggt_pretrained,
                vggt_load_strict=config.vggt_load_strict,
                vggt_enable_track=config.vggt_enable_track,
                freeze_vggt_backbone=config.freeze_vggt_backbone,
            )
        )
        logging.info("VLM2WithPi05 init: VLM2PerceptionModule created in %.2fs", time.perf_counter() - t1)
        
        t2 = time.perf_counter()
        logging.info("VLM2WithPi05 init: creating DualMemoryModule")
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
        logging.info("VLM2WithPi05 init: DualMemoryModule created in %.2fs", time.perf_counter() - t2)
        
        # Pi-0.5 Action Projections (from pi0_pytorch.py)
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)
        
        # Pi-0.5 Time MLP for flow matching
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        
        # Store action config
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        
        # Projection to align 3D representation with language model dimension.
        # If dimensions already match, use identity to avoid perturbing pretrained features.
        if actual_visual_dim == paligemma_config.width:
            self.repr_to_llm: nn.Module = nn.Identity()
            logging.info(
                "VLM2WithPi05 init: using identity repr_to_llm projection (dim=%s)",
                actual_visual_dim,
            )
        else:
            self.repr_to_llm = nn.Linear(actual_visual_dim, paligemma_config.width)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Runtime-only streaming memory state, keyed by session id.
        # This enables websocket servers to share one model instance while keeping
        # per-connection memory isolated.
        self._active_session_id: int | None = None
        self._session_memory_state: dict[int, dict[str, Any]] = {}

    def set_active_session(self, session_id: int | None) -> None:
        """Switch the active streaming-memory session.

        This is a no-op for non-streaming usage, but allows external wrappers to
        isolate memory across concurrent rollouts while reusing one model instance.
        """

        if session_id == self._active_session_id:
            return

        # Save current session runtime buffers before switching away.
        if self._active_session_id is not None:
            self._session_memory_state[self._active_session_id] = self.memory.get_runtime_state()

        self._active_session_id = session_id

        if session_id is None:
            # No session: clear to avoid accidental cross-episode leakage.
            self.memory.clear_runtime_state()
            return

        self.memory.set_runtime_state(self._session_memory_state.get(session_id))

    def reset_streaming_state(self, session_id: int | None = None) -> None:
        """Clear runtime memory buffers for a given session (or current active)."""

        if session_id is None:
            session_id = self._active_session_id

        if session_id is None:
            self.memory.clear_runtime_state()
            return

        self._session_memory_state.pop(session_id, None)
        if session_id == self._active_session_id:
            self.memory.clear_runtime_state()

    def clear_session(self, session_id: int) -> None:
        """Drop stored runtime memory for a session id."""

        self._session_memory_state.pop(session_id, None)
        if session_id == self._active_session_id:
            self._active_session_id = None
            self.memory.clear_runtime_state()
    
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
        *,
        reset_before_process: bool = False,
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

        # Training codepaths may want per-sample memory reset. Streaming inference should not.
        if reset_before_process:
            self.reset_memory(batch_size, device)
        
        all_representations = []
        
        for t in range(num_frames):
            # Get current frame
            frame = video_frames[:, t]  # (batch, C, H, W)
            if frame.dim() == 4 and frame.shape[-1] == 3 and frame.shape[1] != 3:
                frame = frame.permute(0, 3, 1, 2).contiguous()
            
            # Get visual tokens from vision encoder
            visual_tokens = self.paligemma_with_expert.embed_image(frame)  # (batch, n, dim)
            
            # Reshape to spatial format
            n_tokens = visual_tokens.shape[1]
            h = w = int(math.sqrt(n_tokens))
            visual_tokens_spatial = rearrange(visual_tokens, 'b (h w) c -> b h w c', h=h, w=w)
            
            # Apply VLM2 perception (3D-aware representation)
            # Returns H_t = LN(F_pa_t + CrossAttn(F_pa_t, G_vc_t)) per paper Eq (4)
            representation = self.perception(visual_tokens_spatial, frame)
            
            # Apply VLM2 memory with text query
            # Returns M_t = LN(H_t + GatedFusion(W_t, E_t)) per paper Algorithm 1
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
            text_mask=language_masks,
            reset_before_process=True,
        )
        
        # Preserve all frame/camera tokens instead of only the last frame.
        # This keeps prefix token coverage closer to the PI0.5 baseline.
        aggregated_repr = rearrange(memory_enhanced_repr, 'b t n d -> b (t n) d')
        
        # Project to language model dimension
        proj_weight = getattr(self.repr_to_llm, "weight", None)
        proj_dtype = proj_weight.dtype if proj_weight is not None else aggregated_repr.dtype
        prefix_embs = self.repr_to_llm(aggregated_repr.to(proj_dtype))
        
        # Language embeddings are already computed
        lang_emb = lang_emb.to(dtype=prefix_embs.dtype)
        prefix_embs = prefix_embs.to(lang_emb.dtype)
        
        # Combine visual and language embeddings
        prefix_embs = torch.cat([prefix_embs, lang_emb], dim=1)
        
        # Create prefix masks
        n_visual = aggregated_repr.shape[1]
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
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
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
        # Default preprocessing keeps all standard image keys; inference may override.
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
        # For streaming VLA: 3D encoder (VGGT) defaults to head camera only.
        # We also treat each replanning call as one timestep; do NOT build a fake "video" from views.
        processed = cast(
            Any,
            _preprocessing.preprocess_observation_pytorch(observation, train=False, image_keys=("base_0_rgb",)),
        )
        language_tokens = processed.tokenized_prompt
        language_masks = processed.tokenized_prompt_mask
        if language_tokens is None or language_masks is None:
            raise ValueError("Observation missing tokenized_prompt/tokenized_prompt_mask for VLM2 inference.")

        base_image = processed.images["base_0_rgb"]  # (B, C, H, W)
        video_frames = base_image.unsqueeze(1)  # (B, 1, C, H, W)
        batch_size = base_image.shape[0]
        device = base_image.device
        
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

        # Persist runtime memory for the current session (if any).
        if self._active_session_id is not None:
            self._session_memory_state[self._active_session_id] = self.memory.get_runtime_state()
        
        # Aggregate all camera-view tokens (must match training aggregation).
        aggregated_repr = rearrange(memory_enhanced_repr, 'b t n d -> b (t n) d')
        
        # Project to language model dimension
        proj_weight = getattr(self.repr_to_llm, "weight", None)
        proj_dtype = proj_weight.dtype if proj_weight is not None else aggregated_repr.dtype
        prefix_embs = self.repr_to_llm(aggregated_repr.to(proj_dtype))
        
        # Reuse language embeddings
        lang_emb = lang_emb.to(dtype=prefix_embs.dtype)
        prefix_embs = prefix_embs.to(lang_emb.dtype)
        
        # Combine
        prefix_embs = torch.cat([prefix_embs, lang_emb], dim=1)
        
        # Create prefix masks
        n_visual = aggregated_repr.shape[1]
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
        
        # Forward through expert with KV cache.
        # Wrap with PreserveCacheLen so the prefix cache isn't mutated by
        # HF attention layers (they append suffix keys in-place even with
        # use_cache=False).  The prefix cache is reused across denoise steps.
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        with PreserveCacheLen(past_key_values):
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


class VLM2SubtaskWithPi05(VLM2WithPi05):
    def __init__(self, config: VLM2Config, *, alpha: float = 10.0):
        super().__init__(config)
        self.alpha = alpha
        self._text_tokenizer = None
        self._last_predicted_subtasks: list[str] = []

    def make_att_2d_masks(self, pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
        assert make_att_2d_masks is not None
        return make_att_2d_masks(pad_masks, att_masks)

    # ----------------------------------------------------------------------- #
    #  Subtask inference — public API
    # ----------------------------------------------------------------------- #

    def _load_text_tokenizer(self):
        """Lazy-load the PaliGemma sentencepiece tokenizer for subtask decoding."""
        if not hasattr(self, "_text_tokenizer") or self._text_tokenizer is None:
            if not HAS_SENTENCEPIECE:
                raise RuntimeError(
                    "sentencepiece is required for subtask token decoding but is not available."
                )
            assert _download is not None
            assert _sentencepiece is not None
            path = _download.maybe_download(
                "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
            )
            with path.open("rb") as f:
                self._text_tokenizer = _sentencepiece.SentencePieceProcessor(model_proto=f.read())
        return self._text_tokenizer

    def _eos_token_id(self) -> int:
        return self._load_text_tokenizer().eos_id()

    def _has_subtask_conditioning(self, observation) -> bool:
        """Return True if observation already has valid subtask tokens."""
        subtask_mask = getattr(observation, "subtask_mask", None)
        return subtask_mask is not None and bool(torch.any(subtask_mask).item())

    @torch.no_grad()
    def predict_subtask_tokens(
        self,
        observation,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        min_tokens: int = 1,
    ) -> torch.Tensor:
        """Generate subtask token IDs from the current observation.

        Runs the VLM2 perception + memory pipeline to build the visual-language
        prefix, then autoregressively generates subtask text tokens using the
        PaliGemma language model with KV cache.

        Args:
            observation: Model observation (images, state, tokenized prompt).
            max_tokens: Maximum number of subtask tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            min_tokens: Minimum number of tokens before EOS is allowed.

        Returns:
            Integer tensor of shape ``(batch, num_generated)`` with subtask token IDs.
        """
        # Build visual + language prefix the same way sample_actions does,
        # using VLM2's perception + memory pipeline.
        processed = cast(
            Any,
            _preprocessing.preprocess_observation_pytorch(
                observation, train=False, image_keys=("base_0_rgb",)
            ),
        )
        language_tokens = processed.tokenized_prompt
        language_masks = processed.tokenized_prompt_mask
        if language_tokens is None or language_masks is None:
            raise ValueError(
                "Observation missing tokenized_prompt/tokenized_prompt_mask for subtask prediction."
            )

        base_image = processed.images["base_0_rgb"]  # (B, C, H, W)
        video_frames = base_image.unsqueeze(1)  # (B, 1, C, H, W)
        batch_size = base_image.shape[0]
        device = base_image.device

        # Embed language tokens for memory retrieval
        lang_emb = self.paligemma_with_expert.embed_language_tokens(language_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])

        # Save memory state before subtask prediction so we can restore it
        # afterwards.  Gap 3: without this save/restore, hierarchical
        # inference double-updates streaming memory — once during subtask
        # prediction and once during action sampling — corrupting the
        # memory for subsequent timesteps.
        # The subtask prediction is a "read-only" operation: it needs the
        # memory to produce good subtasks, but it must not mutate it.
        # The action-inference path is responsible for the real memory update.
        # Wrapped in try/finally so memory is restored even if
        # process_video_with_memory raises an exception.
        saved_memory_state = self.memory.get_runtime_state()
        try:
            # Process video through VLM2 perception and memory
            memory_enhanced_repr = self.process_video_with_memory(
                video_frames,
                text_query=lang_emb,
                text_mask=language_masks,
            )
        finally:
            # Restore memory state — subtask prediction must not change it,
            # even if the processing above raises an exception.
            self.memory.set_runtime_state(saved_memory_state)

        # NOTE: we do NOT update _session_memory_state here — that is the
        # responsibility of the action-inference path, which performs the
        # real streaming memory update.

        # Aggregate all camera-view tokens (must match training aggregation).
        aggregated_repr = rearrange(memory_enhanced_repr, "b t n d -> b (t n) d")

        # Project to language model dimension
        proj_weight = getattr(self.repr_to_llm, "weight", None)
        proj_dtype = proj_weight.dtype if proj_weight is not None else aggregated_repr.dtype
        prefix_embs = self.repr_to_llm(aggregated_repr.to(proj_dtype))

        # Reuse language embeddings
        lang_emb = lang_emb.to(dtype=prefix_embs.dtype)
        prefix_embs = torch.cat([prefix_embs, lang_emb], dim=1)

        # Create prefix masks
        n_visual = aggregated_repr.shape[1]
        prefix_pad_masks = torch.cat(
            [
                torch.ones(batch_size, n_visual, dtype=torch.bool, device=device),
                language_masks,
            ],
            dim=1,
        )
        prefix_att_masks = torch.zeros(prefix_embs.shape[1], dtype=torch.bool, device=device)
        prefix_att_masks = prefix_att_masks[None, :].expand(batch_size, -1)

        # Create prefix attention masks
        prefix_att_2d_masks = self.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        # Run the prefix through the language model to get initial KV cache.
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        lm_out = self.paligemma_with_expert.paligemma.language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=True,
        )
        prefix_hidden = lm_out.last_hidden_state
        past_kv = lm_out.past_key_values

        embed_weight = self.paligemma_with_expert.paligemma.language_model.embed_tokens.weight
        emb_dim = embed_weight.shape[1]
        lm_head = self.paligemma_with_expert.paligemma.lm_head

        bos_token = self._load_text_tokenizer().bos_id()
        eos_token = self._eos_token_id()
        generated_tokens = []
        next_pos = prefix_pad_masks.sum(dim=-1).to(torch.int64)

        # Track which rows have already produced EOS so we can zero out
        # subsequent tokens in finished rows.  Without this, rows that finish
        # early keep generating garbage tokens because generation only stops
        # when ALL rows have produced EOS (torch.all(...) == eos).
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Inject BOS token as the first step of subtask generation.
        # In training, the subtask sequence is [BOS, tok1, tok2, ..., EOS]
        # with causal attention, and tok1 is predicted from the BOS position.
        # We replicate this at inference by feeding BOS through the model and
        # using its hidden state to predict the first real token.
        bos_token_tensor = torch.full(
            (batch_size, 1), bos_token, dtype=torch.long, device=device
        )
        bos_emb = self.paligemma_with_expert.embed_language_tokens(bos_token_tensor) * math.sqrt(emb_dim)
        # Attention mask: prefix + 1 (BOS)
        bos_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        full_mask_bos = torch.cat([prefix_pad_masks, bos_mask], dim=1)
        full_mask_bos_4d = self._prepare_attention_masks_4d(full_mask_bos[:, None, :])

        bos_out = self.paligemma_with_expert.paligemma.language_model.forward(
            inputs_embeds=bos_emb,
            attention_mask=full_mask_bos_4d,
            position_ids=next_pos[:, None],
            past_key_values=past_kv,
            use_cache=True,
        )
        past_kv = bos_out.past_key_values
        logits = lm_head(bos_out.last_hidden_state)
        next_pos = next_pos + 1

        for step in range(max_tokens):
            if step < min_tokens:
                logits[:, -1, eos_token] = -torch.inf
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1, dtype=torch.float32).to(logits.dtype)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

            # Zero out tokens in rows that have already finished.
            # This prevents garbage post-EOS tokens from contaminating the
            # subtask conditioning seen by the action expert.
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.zeros_like(next_token),
                next_token,
            )

            generated_tokens.append(next_token)

            # Mark rows that just produced EOS as finished (for future steps).
            just_finished = (next_token.squeeze(1) == eos_token) & ~finished
            finished = finished | just_finished

            if torch.all(finished):
                break

            token_emb = self.paligemma_with_expert.embed_language_tokens(next_token) * math.sqrt(emb_dim)
            # Generated tokens so far: 1 (BOS) + step+1 (sampled tokens in loop)
            gen_mask = torch.ones(batch_size, step + 2, dtype=torch.bool, device=device)
            full_mask = torch.cat([prefix_pad_masks, gen_mask], dim=1)
            full_mask_4d = self._prepare_attention_masks_4d(full_mask[:, None, :])

            out = self.paligemma_with_expert.paligemma.language_model.forward(
                inputs_embeds=token_emb,
                attention_mask=full_mask_4d,
                position_ids=next_pos[:, None],
                past_key_values=past_kv,
                use_cache=True,
            )
            last_hidden = out.last_hidden_state
            past_kv = out.past_key_values
            logits = lm_head(last_hidden)
            next_pos = next_pos + 1

        if not generated_tokens:
            return torch.zeros(batch_size, 0, dtype=torch.int32, device=device)
        return torch.cat(generated_tokens, dim=1).to(dtype=torch.int32)

    def decode_subtask_tokens(self, token_batch: torch.Tensor) -> list[str]:
        """Decode a batch of subtask token IDs into human-readable strings.

        Strips BOS, EOS, and padding (zero) tokens before decoding.
        """
        sp = self._load_text_tokenizer()
        bos_token = sp.bos_id()
        eos_token = self._eos_token_id()
        results = []
        for row in token_batch.detach().cpu().tolist():
            tokens = []
            for token in row:
                if token == 0:
                    continue
                if token == bos_token:
                    continue
                if token == eos_token:
                    break
                tokens.append(int(token))
            results.append(sp.decode(tokens))
        return results

    def build_hierarchical_observation(self, observation, subtask_tokens: torch.Tensor):
        """Inject predicted subtask tokens into the observation for action generation.

        Prepends BOS to the predicted subtask tokens so that the action expert
        sees the same conditioning sequence as during training (``[BOS, tok1,
        tok2, ..., EOS]``).  ``predict_subtask_tokens`` returns only ``[tok1,
        ..., EOS]`` (no BOS) because BOS is the generation seed.

        Pads / clips the resulting tokens to ``config.subtask_max_len`` and
        populates the four subtask fields.  ``subtask_loss_mask`` and
        ``subtask_ar_mask`` are zeroed because at inference time the subtask
        is fixed prefix context, not an AR prediction target.

        Args:
            observation: Original model observation.
            subtask_tokens: Generated subtask token IDs (batch, seq_len).
                Must NOT include a leading BOS (output of
                ``predict_subtask_tokens``).

        Returns:
            New Observation with subtask fields populated.
        """
        batch_size = subtask_tokens.shape[0]
        # Default max_len: add 1 to generated token count to accommodate BOS prefix.
        # Without the +1, when config lacks subtask_max_len the last generated
        # token (typically EOS) would be clipped off.
        max_len = getattr(self.config, "subtask_max_len", subtask_tokens.shape[1] + 1)
        device = subtask_tokens.device

        padded_tokens = torch.zeros(batch_size, max_len, dtype=subtask_tokens.dtype, device=device)
        padded_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        padded_loss_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        padded_ar_mask = torch.zeros(batch_size, max_len, dtype=torch.int32, device=device)

        # Prepend BOS token to match training's conditioning sequence.
        # Training uses [BOS, tok1, ..., EOS] as the subtask prefix, and the
        # action expert's cross-attention sees the full sequence including BOS.
        bos_token = self._load_text_tokenizer().bos_id()
        eos_token = self._eos_token_id()
        bos_col = torch.full(
            (batch_size, 1), bos_token, dtype=subtask_tokens.dtype, device=device
        )
        tokens_with_bos = torch.cat([bos_col, subtask_tokens], dim=1)

        # Leave one slot for BOS (subtract 1 from the available max_len).
        clipped = tokens_with_bos[:, :max_len]
        clipped_len = clipped.shape[1]
        padded_tokens[:, :clipped_len] = clipped

        # EOS-aware mask: mark tokens valid from position 0 up to and
        # including the first EOS.  Rows without EOS fall back to != 0.
        # This ensures the action expert never attends to post-EOS garbage
        # tokens in batches where different rows finish at different times.
        #
        # Vectorized implementation (no Python per-row loops):
        #   eos_cumsum[i, j] = number of EOS tokens seen up to position j
        #   valid = cumsum < 1          (before any EOS)
        #         | (cumsum == 1 & is_eos)  (at the first EOS)
        # This correctly handles multiple EOS per row: only the first counts.
        eos_mask_bool = clipped == eos_token
        eos_cumsum = torch.cumsum(eos_mask_bool.to(torch.long), dim=1)  # (batch, clipped_len)
        has_eos = eos_cumsum[:, -1] > 0  # (batch,)

        # Valid positions: before first EOS, OR at the first EOS itself
        valid_up_to_eos = (eos_cumsum < 1) | ((eos_cumsum == 1) & eos_mask_bool)
        # For rows without EOS, use nonzero mask as fallback
        nonzero_mask = clipped != 0
        padded_mask[:, :clipped_len] = torch.where(
            has_eos[:, None], valid_up_to_eos, nonzero_mask
        )

        # During action generation the predicted subtask is fixed context, not an AR target.
        return dataclasses.replace(
            observation,
            subtask_tokens=padded_tokens,
            subtask_mask=padded_mask,
            subtask_loss_mask=padded_loss_mask,
            subtask_ar_mask=padded_ar_mask,
        )

    @torch.no_grad()
    def predict_subtask(
        self, observation, *, max_tokens: int = 64, temperature: float = 0.0
    ) -> list[str]:
        """Generate subtask text strings from observation.

        Convenience wrapper around :meth:`predict_subtask_tokens` +
        :meth:`decode_subtask_tokens`.  Caches the decoded strings on
        ``self._last_predicted_subtasks`` for the Policy wrapper to pick up.
        """
        generated_tokens = self.predict_subtask_tokens(
            observation, max_tokens=max_tokens, temperature=temperature
        )
        results = self.decode_subtask_tokens(generated_tokens)
        self._last_predicted_subtasks = results
        return results

    @torch.no_grad()
    def sample_actions_hierarchical(
        self,
        device: torch.device,
        observation,
        noise: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        *,
        max_subtask_tokens: int = 64,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """End-to-end hierarchical inference: predict subtask, then sample actions.

        1. Generate subtask tokens from the observation.
        2. Build a hierarchical observation conditioned on the predicted subtask.
        3. Sample actions using the subtask-conditioned prefix.

        Args:
            device: Torch device.
            observation: Model observation.
            noise: Optional initial noise for flow matching.
            num_steps: Number of Euler steps for flow matching.
            max_subtask_tokens: Maximum subtask tokens to generate.
            temperature: Subtask sampling temperature.

        Returns:
            Sampled actions ``(batch, action_horizon, action_dim)``.
        """
        generated_tokens = self.predict_subtask_tokens(
            observation, max_tokens=max_subtask_tokens, temperature=temperature
        )
        self._last_predicted_subtasks = self.decode_subtask_tokens(generated_tokens)
        conditioned_observation = self.build_hierarchical_observation(observation, generated_tokens)
        return self._sample_actions_with_subtask_conditioning(
            device, conditioned_observation, noise=noise, num_steps=num_steps
        )

    @torch.no_grad()
    def sample_actions(
        self,
        device: torch.device,
        observation,
        noise: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        *,
        max_subtask_tokens: int = 64,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Sample actions, dispatching to subtask-conditioned or hierarchical path.

        - If the observation already has subtask conditioning (``subtask_mask``
          has True entries), use the subtask-conditioned path directly.
        - Otherwise, generate a subtask first (hierarchical inference).

        This matches the behaviour of :class:`PI05SubtaskPytorch.sample_actions`
        so that the Policy wrapper sees a uniform interface.
        """
        if self._has_subtask_conditioning(observation):
            return self._sample_actions_with_subtask_conditioning(
                device, observation, noise=noise, num_steps=num_steps
            )
        return self.sample_actions_hierarchical(
            device,
            observation,
            noise=noise,
            num_steps=num_steps,
            max_subtask_tokens=max_subtask_tokens,
            temperature=temperature,
        )

    # ----------------------------------------------------------------------- #
    #  Internal helpers
    # ----------------------------------------------------------------------- #

    @torch.no_grad()
    def _sample_actions_with_subtask_conditioning(
        self,
        device: torch.device,
        observation,
        noise: Optional[torch.Tensor] = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Sample actions with subtask tokens included in the prefix.

        Same structure as the base class :meth:`VLM2WithPi05.sample_actions`,
        but inserts subtask embeddings between the language prefix and the
        action suffix so the action decoder sees the predicted subtask as
        additional context.

        Args:
            device: Torch device.
            observation: Observation with subtask fields populated.
            noise: Optional initial noise.
            num_steps: Number of denoising steps.

        Returns:
            Sampled actions ``(batch, action_horizon, action_dim)``.
        """
        processed = cast(
            Any,
            _preprocessing.preprocess_observation_pytorch(
                observation, train=False, image_keys=("base_0_rgb",)
            ),
        )
        language_tokens = processed.tokenized_prompt
        language_masks = processed.tokenized_prompt_mask
        if language_tokens is None or language_masks is None:
            raise ValueError(
                "Observation missing tokenized_prompt/tokenized_prompt_mask "
                "for subtask-conditioned action sampling."
            )

        subtask_tokens = observation.subtask_tokens
        subtask_mask = observation.subtask_mask
        if subtask_tokens is None or subtask_mask is None:
            raise ValueError(
                "_sample_actions_with_subtask_conditioning called but subtask_tokens/mask is None."
            )

        base_image = processed.images["base_0_rgb"]  # (B, C, H, W)
        video_frames = base_image.unsqueeze(1)  # (B, 1, C, H, W)
        batch_size = base_image.shape[0]
        device = base_image.device

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
            text_mask=language_masks,
        )

        # Persist runtime memory for the current session (if any).
        if self._active_session_id is not None:
            self._session_memory_state[self._active_session_id] = self.memory.get_runtime_state()

        # Aggregate all camera-view tokens (must match training aggregation).
        aggregated_repr = rearrange(memory_enhanced_repr, "b t n d -> b (t n) d")

        # Project to language model dimension
        proj_weight = getattr(self.repr_to_llm, "weight", None)
        proj_dtype = proj_weight.dtype if proj_weight is not None else aggregated_repr.dtype
        prefix_embs = self.repr_to_llm(aggregated_repr.to(proj_dtype))

        # Append language embeddings
        lang_emb = lang_emb.to(dtype=prefix_embs.dtype)
        prefix_embs = torch.cat([prefix_embs, lang_emb], dim=1)

        n_visual = aggregated_repr.shape[1]
        prefix_pad_masks = torch.cat(
            [
                torch.ones(batch_size, n_visual, dtype=torch.bool, device=device),
                language_masks,
            ],
            dim=1,
        )
        prefix_att_masks = torch.zeros(prefix_embs.shape[1], dtype=torch.bool, device=device)
        prefix_att_masks = prefix_att_masks[None, :].expand(batch_size, -1)

        # Append subtask embeddings to the prefix.
        subtask_embs = self.paligemma_with_expert.embed_language_tokens(subtask_tokens)
        subtask_embs = subtask_embs * math.sqrt(subtask_embs.shape[-1])
        subtask_embs = subtask_embs.to(dtype=prefix_embs.dtype)
        prefix_embs = torch.cat([prefix_embs, subtask_embs], dim=1)

        prefix_pad_masks = torch.cat([prefix_pad_masks, subtask_mask], dim=1)
        prefix_att_masks = torch.cat(
            [
                prefix_att_masks,
                torch.ones_like(subtask_mask, dtype=prefix_att_masks.dtype),
            ],
            dim=1,
        )

        # Build prefix attention masks and KV cache
        prefix_att_2d_masks = self.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

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
            v_t = self._denoise_step_subtask(
                x_t, expanded_time, prefix_pad_masks, past_key_values
            )
            x_t = x_t + dt * v_t
            time = time + dt

        return x_t

    def _denoise_step_subtask(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Any,
    ) -> torch.Tensor:
        """Single denoising step (subtask-conditioned version).

        Mirrors :meth:`VLM2WithPi05._denoise_step` but uses the expert's
        forward with KV cache that already includes the subtask context.
        The logic is identical — the difference is entirely in how the
        prefix KV cache was built.  Kept as a separate method for clarity
        and to avoid any hidden coupling with the base implementation.
        """
        # Embed action suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_action_suffix(x_t, timestep)

        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]

        # Create attention masks: suffix tokens can attend to all prefix tokens
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = self.make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # Position IDs
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        # Forward through expert with KV cache.
        # Wrap with PreserveCacheLen so the prefix cache isn't mutated by
        # HF attention layers (they append suffix keys in-place even with
        # use_cache=False).  The prefix cache is reused across denoise steps.
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        with PreserveCacheLen(past_key_values):
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
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        return self.action_out_proj(suffix_out)

    def forward(
        self,
        video_frames: torch.Tensor,
        point_maps: torch.Tensor,
        language_tokens: torch.Tensor,
        language_masks: torch.Tensor,
        actions: torch.Tensor,
        *,
        subtask_tokens: torch.Tensor | None = None,
        subtask_mask: torch.Tensor | None = None,
        subtask_ar_mask: torch.Tensor | None = None,
        subtask_loss_mask: torch.Tensor | None = None,
        noise: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = actions.shape[0]
        device = actions.device

        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(batch_size, device)
        assert time is not None

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        lang_emb = self.paligemma_with_expert.embed_language_tokens(language_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])

        memory_enhanced_repr = self.process_video_with_memory(
            video_frames,
            text_query=lang_emb,
            text_mask=language_masks,
            reset_before_process=True,
        )
        aggregated_repr = rearrange(memory_enhanced_repr, "b t n d -> b (t n) d")

        proj_weight = getattr(self.repr_to_llm, "weight", None)
        proj_dtype = proj_weight.dtype if proj_weight is not None else aggregated_repr.dtype
        visual_prefix = self.repr_to_llm(aggregated_repr.to(proj_dtype))

        lang_emb = lang_emb.to(dtype=visual_prefix.dtype)
        prefix_embs = torch.cat([visual_prefix, lang_emb], dim=1)

        n_visual = aggregated_repr.shape[1]
        prefix_pad_masks = torch.cat(
            [
                torch.ones(batch_size, n_visual, dtype=torch.bool, device=device),
                language_masks,
            ],
            dim=1,
        )
        prefix_att_masks = torch.zeros(prefix_embs.shape[1], dtype=torch.bool, device=device)
        prefix_att_masks = prefix_att_masks[None, :].expand(batch_size, -1)

        has_subtask = (
            subtask_tokens is not None
            and subtask_mask is not None
            and subtask_loss_mask is not None
            and bool(subtask_loss_mask.any().item())
        )

        prefix_len_no_subtask = prefix_embs.shape[1]
        if has_subtask:
            subtask_embs = self.paligemma_with_expert.embed_language_tokens(subtask_tokens)
            subtask_embs = subtask_embs * math.sqrt(subtask_embs.shape[-1])
            prefix_embs = torch.cat([prefix_embs, subtask_embs], dim=1)
            prefix_pad_masks = torch.cat([prefix_pad_masks, subtask_mask], dim=1)
            prefix_att_masks = torch.cat(
                [prefix_att_masks, torch.ones_like(subtask_mask, dtype=prefix_att_masks.dtype)],
                dim=1,
            )

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_action_suffix(x_t, time)

        prefix_embs, suffix_embs = align_tensors_to_reference_dtype(
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight,
            prefix_embs,
            suffix_embs,
            context="language model",
        )
        prefix_att_masks = prefix_att_masks.to(dtype=suffix_att_masks.dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = self.make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        if suffix_out is None:
            raise RuntimeError("Expected suffix outputs from PaliGemma expert forward pass.")

        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t_pred = self.action_out_proj(suffix_out)
        flow_loss = F.mse_loss(v_t_pred.float(), u_t.float(), reduction="mean")

        if not has_subtask:
            return {
                "loss": flow_loss,
                "flow_loss": flow_loss.detach(),
                "ce_loss": torch.tensor(0.0, device=device),
            }

        assert prefix_out is not None
        assert subtask_tokens is not None
        assert subtask_loss_mask is not None

        subtask_len = subtask_tokens.shape[1]
        subtask_hidden = prefix_out[:, prefix_len_no_subtask : prefix_len_no_subtask + subtask_len]

        subtask_hidden = subtask_hidden.to(
            dtype=self.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.dtype
        )
        text_logits = torch.matmul(
            subtask_hidden,
            self.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.T,
        )

        shift_logits = text_logits[:, :-1].contiguous()
        shift_targets = subtask_tokens[:, 1:].contiguous().to(dtype=torch.long)
        shift_loss_mask = subtask_loss_mask[:, 1:].contiguous().float()

        ce_loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction="none",
        ).view(shift_logits.shape[0], -1)

        # Global mean CE loss over all valid tokens in the batch.
        # Previously we computed per-sample mean then averaged across batch (mean-of-means),
        # which amplified variance when batch composition varied (e.g., some samples had no
        # valid subtask tokens, producing ce_loss=0 that skewed the batch mean).
        # Now we sum all per-token losses and divide by total valid tokens directly,
        # which is the standard NLP practice for consistent loss statistics.
        total_ce_loss = (ce_loss_per_token * shift_loss_mask).sum()
        total_valid_tokens = shift_loss_mask.sum().clamp(min=1)
        ce_loss = total_ce_loss / total_valid_tokens

        combined_loss = ce_loss + self.alpha * flow_loss
        return {
            "loss": combined_loss,
            "flow_loss": flow_loss.detach(),
            "ce_loss": ce_loss.detach(),
        }


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
