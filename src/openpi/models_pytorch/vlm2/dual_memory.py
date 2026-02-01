"""Dual-Memory Module for VLM2.

This module implements the dual-memory architecture from VLM2:
1. Working Memory - sliding window for immediate context
2. Episodic Memory - fixed-capacity bank for long-term storage
3. Gated Memory Fusion - combines both memory streams

Reference: VLM2 paper Section 3.3
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class QueryFusion(nn.Module):
    """Fuses visual representation and text query for memory retrieval.
    
    Q = CrossAttn(Visual, Text)
    
    Args:
        feature_dim: Dimension of features
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: (batch, n_vis, dim)
            text_tokens: (batch, n_text, dim)
            text_mask: (batch, n_text) boolean mask (True=keep, False=mask)
            
        Returns:
            Fused query tokens (batch, n_vis, dim)
        """
        # Cross-Attention: Query=Visual, Key=Text, Value=Text
        # We want to enhance visual query with text information.
        
        key_padding_mask = None
        if text_mask is not None:
            # MultiheadAttention expects True for padding (ignore)
            # Input text_mask is usually True for valid tokens.
            key_padding_mask = ~text_mask
            
        fused, _ = self.attention(
            query=visual_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=key_padding_mask,
        )
        
        # Residual + Norm
        output = self.layer_norm(visual_tokens + fused)
        return output


class MemoryAttention(nn.Module):
    """Cross-attention module for memory retrieval.
    
    Used by both Working Memory and Episodic Memory for retrieval.
    
    Args:
        query_dim: Dimension of query features
        memory_dim: Dimension of memory features
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        query_dim: int,
        memory_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        
        assert query_dim % num_heads == 0, \
            f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})"
        
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(memory_dim, query_dim)
        self.v_proj = nn.Linear(memory_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform cross-attention between query and memory.
        
        Args:
            query: Query features (batch, n_query, query_dim)
            memory: Memory features (batch, n_memory, memory_dim)
            attention_mask: Optional mask (batch, n_query, n_memory)
            
        Returns:
            Retrieved features (batch, n_query, query_dim)
        """
        batch_size, n_query, _ = query.shape
        _, n_memory, _ = memory.shape

        original_dtype = query.dtype
        target_dtype = self.q_proj.weight.dtype
        query = query.to(target_dtype)
        memory = memory.to(target_dtype)
        
        # Compute Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(memory)
        v = self.v_proj(memory)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn.masked_fill(~attention_mask.unsqueeze(1), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.out_proj(out)
        return out.to(original_dtype)


class WorkingMemory(nn.Module):
    """Working Memory module with sliding window.
    
    Maintains a sliding window buffer of the most recent Lw representations,
    allowing for immediate context retrieval.
    
    Reference: VLM2 paper Section 3.3 - Working Memory for Immediate Retrieval
    
    Args:
        feature_dim: Dimension of stored features
        window_size: Maximum number of entries (Lw)
        num_heads: Number of attention heads for retrieval
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        feature_dim: int,
        window_size: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        
        # Cross-attention for retrieval
        self.attention = MemoryAttention(
            query_dim=feature_dim,
            memory_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Memory buffer (not a parameter, managed externally)
        self.register_buffer("memory_buffer", None, persistent=False)
        self.register_buffer("memory_count", torch.tensor(0), persistent=False)
        self.register_buffer("num_tokens", None, persistent=False)
    
    def reset(self, batch_size: int, device: torch.device, num_tokens: int | None = None):
        """Reset memory buffer for new sequences.
        
        Args:
            batch_size: Batch size for the new sequence
            device: Device to allocate memory on
        """
        if num_tokens is None:
            self.memory_buffer = None
            self.num_tokens = None
        else:
            self.memory_buffer = torch.zeros(
                batch_size, self.window_size, num_tokens, self.feature_dim,
                device=device,
            )
            self.num_tokens = torch.tensor(num_tokens, device=device)
        self.memory_count = torch.tensor(0, device=device)
    
    def add(self, features: torch.Tensor):
        """Add new features to the working memory.
        
        Uses FIFO policy to maintain the sliding window.
        
        Args:
            features: Features to add (batch, n, feature_dim)
        """
        batch_size, n_tokens, _ = features.shape

        if self.memory_buffer is None or (self.num_tokens is not None and self.num_tokens.item() != n_tokens):
            self.reset(batch_size, features.device, num_tokens=n_tokens)
        if self.memory_buffer is None:
            raise RuntimeError("Working memory buffer was not initialized.")

        # Add to buffer with FIFO policy (store token-level features)
        if self.memory_count.item() < self.window_size:
            idx = self.memory_count.item()
            updated_buffer = self.memory_buffer.clone()
            updated_buffer[:, idx] = features
            self.memory_buffer = updated_buffer
            self.memory_count = self.memory_count + 1
        else:
            # Shift buffer and add to end
            rolled_buffer = torch.roll(self.memory_buffer, shifts=-1, dims=1)
            rolled_buffer[:, -1] = features
            self.memory_buffer = rolled_buffer
    
    def get_memory(self) -> torch.Tensor:
        """Get current memory contents.
        
        Returns:
            Memory contents (batch, current_size, n_tokens, feature_dim)
        """
        if self.memory_buffer is None or self.memory_count.item() == 0:
            return None
        
        return self.memory_buffer[:, :self.memory_count]
    
    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve from working memory using cross-attention.
        
        M_w_t = Attn(Q(Ht), K(Wt), V(Wt))
        
        Args:
            query: Query features Ht (batch, n, feature_dim)
            
        Returns:
            Retrieved features M_w_t (batch, n, feature_dim)
        """
        memory = self.get_memory()

        if memory is None:
            # Return query as-is if memory is empty
            return query

        # Flatten memory tokens across time: (batch, window*n_tokens, feature_dim)
        batch_size, num_steps, num_tokens, dim = memory.shape
        memory_flat = memory.reshape(batch_size, num_steps * num_tokens, dim)

        return self.attention(query, memory_flat)


class EpisodicMemory(nn.Module):
    """Episodic Memory module with fixed capacity.
    
    Maintains a fixed-capacity bank for long-term storage of salient observations.
    Uses similarity-based update to maintain diversity.
    
    Reference: VLM2 paper Section 3.3 - Episodic Memory for Long-Horizon Recall
    
    Args:
        feature_dim: Dimension of stored features
        capacity: Maximum number of entries (Le)
        num_heads: Number of attention heads for retrieval
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        feature_dim: int,
        capacity: int = 32,
        num_heads: int = 8,
        dropout: float = 0.0,
        similarity_threshold: float = 0.7,
        fusion_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.capacity = capacity
        
        # Cross-attention for retrieval
        self.attention = MemoryAttention(
            query_dim=feature_dim,
            memory_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Memory bank (not a parameter, managed externally)
        self.register_buffer("memory_bank", None, persistent=False)
        self.register_buffer("memory_count", torch.tensor(0), persistent=False)
        self.register_buffer("num_tokens", None, persistent=False)
        self.register_buffer("last_used", None, persistent=False)
        self.register_buffer("step", torch.tensor(0), persistent=False)

        self.similarity_threshold = similarity_threshold
        self.fusion_alpha = fusion_alpha
    
    def reset(self, batch_size: int, device: torch.device, num_tokens: int | None = None):
        """Reset memory bank for new sequences.
        
        Args:
            batch_size: Batch size for the new sequence
            device: Device to allocate memory on
        """
        if num_tokens is None:
            self.memory_bank = None
            self.num_tokens = None
            self.last_used = None
        else:
            self.memory_bank = torch.zeros(
                batch_size, self.capacity, num_tokens, self.feature_dim,
                device=device,
            )
            self.num_tokens = torch.tensor(num_tokens, device=device)
            self.last_used = torch.zeros(batch_size, self.capacity, device=device, dtype=torch.long)
        self.memory_count = torch.tensor(0, device=device)
        self.step = torch.tensor(0, device=device)
    
    def add(self, features: torch.Tensor):
        """Add new features to episodic memory.
        
        Uses similarity-based replacement to maintain diversity when at capacity.
        i*_t = argmax_i cos(Mt, Ei)
        
        Args:
            features: Features to add (batch, n, feature_dim)
        """
        batch_size, n_tokens, _ = features.shape

        if self.memory_bank is None or (self.num_tokens is not None and self.num_tokens.item() != n_tokens):
            self.reset(batch_size, features.device, num_tokens=n_tokens)
        if self.memory_bank is None or self.last_used is None:
            raise RuntimeError("Episodic memory bank was not initialized.")

        self.step = self.step + 1
        updated_bank = self.memory_bank.clone()
        updated_last_used = self.last_used.clone()

        # Aggregate features for similarity computation
        aggregated = features.mean(dim=1)  # (batch, feature_dim)

        if self.memory_count.item() < self.capacity:
            # Add to next available slot
            idx = self.memory_count.item()
            updated_bank[:, idx] = features
            updated_last_used[:, idx] = self.step
            self.memory_bank = updated_bank
            self.last_used = updated_last_used
            self.memory_count = self.memory_count + 1
            return

        # Memory full: compute similarity and decide update strategy
        bank_pooled = updated_bank.mean(dim=2)  # (batch, capacity, feature_dim)
        normalized_new = F.normalize(aggregated, dim=-1)
        normalized_bank = F.normalize(bank_pooled, dim=-1)
        similarities = torch.einsum("bd,bnd->bn", normalized_new, normalized_bank)  # (batch, capacity)
        max_sim, most_similar_idx = similarities.max(dim=1)

        batch_indices = torch.arange(batch_size, device=features.device)
        should_merge = max_sim > self.similarity_threshold

        if should_merge.any():
            merge_indices = most_similar_idx[should_merge]
            merge_batches = batch_indices[should_merge]
            existing = updated_bank[merge_batches, merge_indices]
            updated = self.fusion_alpha * existing + (1 - self.fusion_alpha) * features[merge_batches]
            updated_bank[merge_batches, merge_indices] = updated
            updated_last_used[merge_batches, merge_indices] = self.step

        if (~should_merge).any():
            replace_batches = batch_indices[~should_merge]
            # Replace least recently used entries
            lru_indices = updated_last_used[replace_batches].argmin(dim=1)
            updated_bank[replace_batches, lru_indices] = features[replace_batches]
            updated_last_used[replace_batches, lru_indices] = self.step

        self.memory_bank = updated_bank
        self.last_used = updated_last_used
    
    def get_memory(self) -> torch.Tensor:
        """Get current memory contents.
        
        Returns:
            Memory contents (batch, current_size, n_tokens, feature_dim)
        """
        if self.memory_bank is None or self.memory_count.item() == 0:
            return None
        
        return self.memory_bank[:, :self.memory_count]
    
    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve from episodic memory using cross-attention.
        
        M_e_t = Attn(Q(Ht), K(Et), V(Et))
        
        Args:
            query: Query features Ht (batch, n, feature_dim)
            
        Returns:
            Retrieved features M_e_t (batch, n, feature_dim)
        """
        memory = self.get_memory()

        if memory is None:
            # Return query as-is if memory is empty
            return query

        batch_size, num_entries, num_tokens, dim = memory.shape
        memory_flat = memory.reshape(batch_size, num_entries * num_tokens, dim)

        return self.attention(query, memory_flat)


class GatedMemoryFusion(nn.Module):
    """Gated fusion of working and episodic memory.
    
    Combines information from both memory streams using a learnable gate.
    
    γt = σ(MLP(Concat[M_w_t; M_e_t]))
    Mt = γt ⊙ M_w_t + (1 - γt) ⊙ M_e_t
    
    Reference: VLM2 paper Section 3.3 - Memory Fusion and Update
    
    Args:
        feature_dim: Dimension of features
        hidden_dim: Hidden dimension for gate MLP
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim or feature_dim
        
        # Gate network: MLP that outputs scalar gate per position
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, feature_dim),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        working_memory_output: torch.Tensor,
        episodic_memory_output: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse working and episodic memory outputs with gating.
        
        Args:
            working_memory_output: M_w_t (batch, n, feature_dim)
            episodic_memory_output: M_e_t (batch, n, feature_dim)
            
        Returns:
            Fused memory output Mt (batch, n, feature_dim)
        """
        original_dtype = working_memory_output.dtype
        target_dtype = self.gate_mlp[0].weight.dtype
        working_memory_output = working_memory_output.to(target_dtype)
        episodic_memory_output = episodic_memory_output.to(target_dtype)

        # Concatenate for gate computation
        concat_features = torch.cat([working_memory_output, episodic_memory_output], dim=-1)

        # Compute gate: γt = σ(MLP(Concat[M_w_t; M_e_t]))
        gate = self.gate_mlp(concat_features)  # (batch, n, feature_dim)

        # Gated fusion: Mt = γt ⊙ M_w_t + (1 - γt) ⊙ M_e_t
        fused = gate * working_memory_output + (1 - gate) * episodic_memory_output
        fused = fused.to(original_dtype)
        
        return fused


class DualMemoryModule(nn.Module):
    """Complete Dual-Memory Module.
    
    Combines Working Memory and Episodic Memory with gated fusion
    for persistent spatial reasoning.
    
    Reference: VLM2 paper Section 3.3
    
    Args:
        feature_dim: Dimension of features
        working_memory_size: Size of working memory window (Lw)
        episodic_memory_capacity: Capacity of episodic memory (Le)
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension for MLPs
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        feature_dim: int,
        working_memory_size: int = 8,
        episodic_memory_capacity: int = 32,
        num_heads: int = 8,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        similarity_threshold: float = 0.7,
        fusion_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Working Memory (sliding window)
        self.working_memory = WorkingMemory(
            feature_dim=feature_dim,
            window_size=working_memory_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Episodic Memory (fixed capacity)
        self.episodic_memory = EpisodicMemory(
            feature_dim=feature_dim,
            capacity=episodic_memory_capacity,
            num_heads=num_heads,
            dropout=dropout,
            similarity_threshold=similarity_threshold,
            fusion_alpha=fusion_alpha,
        )
        
        # Gated fusion
        self.fusion = GatedMemoryFusion(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )
        
        # Query Fusion (Visual + Text)
        self.query_fusion = QueryFusion(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Layer norm for output
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def reset(self, batch_size: int, device: torch.device):
        """Reset all memory components.
        
        Args:
            batch_size: Batch size for the new sequence
            device: Device to allocate memory on
        """
        self.working_memory.reset(batch_size, device)
        self.episodic_memory.reset(batch_size, device)
    
    def forward(
        self,
        current_representation: torch.Tensor,
        text_query: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """Process current representation through dual-memory.
        
        Implements Algorithm 1 from the paper.
        
        Args:
            current_representation: 3D-aware representation Ht (batch, n, feature_dim)
            text_query: Optional text instruction embeddings (batch, len, feature_dim)
            text_mask: Optional text mask (batch, len)
            update_memory: Whether to update memory with current representation
            
        Returns:
            Memory-enhanced representation Mt (batch, n, feature_dim)
        """
        batch_size = current_representation.shape[0]
        device = current_representation.device
        
        # Initialize memory if needed
        if self.working_memory.memory_buffer is None:
            self.reset(batch_size, device)
            
        # Form retrieval query
        if text_query is not None:
            retrieval_query = self.query_fusion(current_representation, text_query, text_mask)
        else:
            retrieval_query = current_representation
        
        # 1. Working Memory Retrieval
        # M_w_t = Attn(Q(Ht), K(Wt), V(Wt))
        working_output = self.working_memory.retrieve(retrieval_query)
        
        # 2. Episodic Memory Retrieval
        # M_e_t = Attn(Q(Ht), K(Et), V(Et))
        episodic_output = self.episodic_memory.retrieve(retrieval_query)
        
        # 3. Gated Memory Fusion
        # γt = σ(MLP(Concat[M_w_t; M_e_t]))
        # Mt = γt ⊙ M_w_t + (1 - γt) ⊙ M_e_t
        fused_output = self.fusion(working_output, episodic_output)
        
        # Apply residual connection and normalization
        original_dtype = current_representation.dtype
        target_dtype = self.layer_norm.weight.dtype
        # Note: we add memory output to the ORIGNAL representation, not the query
        current_representation = current_representation.to(target_dtype)
        fused_output = fused_output.to(target_dtype)
        output = self.layer_norm(current_representation + fused_output)
        output = output.to(original_dtype)
        
        if update_memory:
            # 4. Update Working Memory (FIFO sliding window)
            # Store the original representation? Or memory-enhanced?
            # VLM2 paper: store Ht (3D-aware representation)
            self.working_memory.add(current_representation)
            
            # 5. Update Episodic Memory (similarity-based replacement)
            # Paper says store "salient observations".
            # "We store Ht in episodic memory"
            self.episodic_memory.add(current_representation)
            
            # Note: The original code stored 'output' (memory enhanced) in episodic memory.
            # "self.episodic_memory.add(output)" was line 571.
            # Storing enhanced representation might lead to drift?
            # Storing raw 3D-aware representation (current_representation) seems safer and cleaner.
            # Changed to current_representation.
        
        return output
    
    def get_memory_stats(self) -> dict:
        """Get current memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            'working_memory_count': self.working_memory.memory_count.item() 
                if self.working_memory.memory_count is not None else 0,
            'working_memory_capacity': self.working_memory.window_size,
            'episodic_memory_count': self.episodic_memory.memory_count.item()
                if self.episodic_memory.memory_count is not None else 0,
            'episodic_memory_capacity': self.episodic_memory.capacity,
            'episodic_similarity_threshold': self.episodic_memory.similarity_threshold,
            'episodic_fusion_alpha': self.episodic_memory.fusion_alpha,
        }
