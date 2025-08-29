import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.transformer import _get_activation_fn # If needed for custom layers

# --- PositionalEncoding remains the same ---
class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the token embeddings for the Transformer
    """
    def __init__(self, d_model, max_seq_length=5000, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Inherits but overrides forward completely to inject bias correctly
class DistanceAwareMultiheadAttention(nn.MultiheadAttention):
    """
    Extends MultiheadAttention to incorporate distance-based attention bias *before* softmax.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, # Added bias param
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first=True, device=None, dtype=None):
        # Call parent __init__ BUT we will mostly ignore its internal projection setup
        # We call it for attribute initialization (embed_dim, num_heads, etc.)
        # Note: We explicitly set batch_first=True here, overriding potential parent defaults if different
        super(DistanceAwareMultiheadAttention, self).__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
            kdim, vdim, batch_first, device, dtype)

        # Re-define projection layers explicitly for clarity and control
        # This overrides the internal complex projection logic of nn.MultiheadAttention
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.kdim, self.embed_dim, bias=bias, device=device, dtype=dtype) # Use kdim
        self.v_proj = nn.Linear(self.vdim, self.embed_dim, bias=bias, device=device, dtype=dtype) # Use vdim
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, device=device, dtype=dtype)

        # Add a learnable scaling factor for distance influence
        self.distance_scaling = nn.Parameter(torch.tensor(1.0))
        self._reset_parameters_explicit() # Initialize our explicit layers

    def _reset_parameters_explicit(self):
        # Initialize weights for our explicitly defined layers
        nn.init.xavier_uniform_(self.q_proj.weight)
        if self.q_proj.bias is not None: nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.xavier_uniform_(self.k_proj.weight)
        if self.k_proj.bias is not None: nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if self.v_proj.bias is not None: nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None: nn.init.constant_(self.out_proj.bias, 0.)


    def forward(self, query, key, value, distances=None,
                key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True):
        """
        Forward pass with distance-based attention bias applied *before* softmax.

        Args:
            query, key, value: Input tensors. Shape depends on batch_first.
            distances: [batch_size, seq_len_q, seq_len_k] tensor of distances.
                       Smaller values = closer. Assumed pre-computed.
            key_padding_mask: Mask for key sequence. Shape [batch_size, seq_len_k].
            attn_mask: Additive mask for attention scores. Can be 2D or 3D.
                       Shape [seq_len_q, seq_len_k] or [batch_size * num_heads, seq_len_q, seq_len_k].
            need_weights: If True, returns attention weights.
            average_attn_weights: If True and need_weights=True, averages attention weights
                                  across heads. Otherwise returns per-head weights.

        Returns:
            attn_output: Output tensor. Shape [batch_size, seq_len_q, embed_dim] if batch_first.
            attn_output_weights: Attention weights. Shape depends on need_weights and
                                 average_attn_weights. E.g., [batch_size, seq_len_q, seq_len_k]
                                 if averaged, or [batch_size, num_heads, seq_len_q, seq_len_k]
                                 if not averaged.
        """
        if self.batch_first:
            # Ensure inputs are [batch_size, seq_len, embed_dim]
            query, key, value = [x if x.dim() == 3 else x.unsqueeze(0) for x in (query, key, value)] # Basic check
        else:
             # Convert to [batch_size, seq_len, embed_dim] internally for easier processing
             query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)

        # --- 1. Linear Projections ---
        q = self.q_proj(query) # [batch_size, seq_len_q, embed_dim]
        k = self.k_proj(key)   # [batch_size, seq_len_k, embed_dim]
        v = self.v_proj(value) # [batch_size, seq_len_v, embed_dim]

        # --- 2. Reshape for Multi-Head ---
        # Scale q (standard practice, though sometimes done after score calc)
        q = q * (self.head_dim ** -0.5)
        # Reshape: [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim] -> [batch_size * num_heads, seq_len, head_dim]
        q = q.contiguous().view(batch_size, seq_len_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, seq_len_q, self.head_dim)
        k = k.contiguous().view(batch_size, seq_len_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, seq_len_k, self.head_dim)
        v = v.contiguous().view(batch_size, seq_len_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, seq_len_v, self.head_dim)

        # --- 3. Calculate Raw Attention Scores ---
        # (B * H, Lq, D) @ (B * H, Lk, D).transpose -> (B * H, Lq, Lk)
        attn_scores = torch.bmm(q, k.transpose(1, 2))

        # --- 4. Inject Distance Bias (Additive, Before Softmax) ---
        if distances is not None:
            # distances shape: [batch_size, seq_len_q, seq_len_k]
            # Additive bias: closer = higher score => use -log(dist) or similar
            # Add epsilon for numerical stability if distances can be zero
            distance_bias = -torch.log(distances + 1e-9) * self.distance_scaling.abs() # Use abs() scaling
            # Expected bias shape: [batch_size, seq_len_q, seq_len_k]

            # Expand bias for heads: [batch_size, seq_len_q, seq_len_k] -> [batch_size, 1, seq_len_q, seq_len_k] -> [batch_size, num_heads, seq_len_q, seq_len_k] -> [batch_size * num_heads, seq_len_q, seq_len_k]
            
            distance_bias = distance_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

            seq_len_q = distance_bias.size(2)
            seq_len_k = distance_bias.size(3)

            distance_bias = distance_bias.view(batch_size * self.num_heads, seq_len_q, seq_len_k)

            # Assuming distance_bias has shape [B, H, 50, 22]
            padding = torch.zeros((batch_size * self.num_heads, seq_len_q, seq_len_q - seq_len_k), device=distance_bias.device)
            distance_bias_padded = torch.cat([distance_bias, padding], dim=-1)  # Now shape [B, H, 50, 50]

            attn_scores = attn_scores + distance_bias_padded # Add bias BEFORE softmax

        # --- 5. Apply Masks (Additive attn_mask, Boolean key_padding_mask) ---
        if attn_mask is not None:
             # Ensure attn_mask is broadcastable: [Lq, Lk], [H, Lq, Lk], or [B*H, Lq, Lk]
             if attn_mask.dim() == 2:
                 attn_mask = attn_mask.unsqueeze(0) # Add head/batch dim
             # Check if needs further expansion for batch*heads
             if attn_mask.size(0) != attn_scores.size(0):
                  attn_mask = attn_mask.repeat(batch_size, 1, 1) # Basic repeat for heads if needed, adjust if more complex

             attn_scores = attn_scores + attn_mask # Additive mask

        if key_padding_mask is not None:
            # key_padding_mask shape: [batch_size, seq_len_k]
            # Expand to [B*H, Lq, Lk]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, Lk]
            mask = mask.repeat(1, self.num_heads, seq_len_q, 1) # [B, H, Lq, Lk]
            mask = mask.view(batch_size * self.num_heads, seq_len_q, seq_len_k) # [B*H, Lq, Lk]
            # Apply mask: set score to -inf where mask is True
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # --- 6. Softmax ---
        attn_weights = F.softmax(attn_scores, dim=-1) # [B*H, Lq, Lk]

        # --- 7. Dropout ---
        attn_weights_dropped = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # --- 8. Apply Attention to Values ---
        # (B*H, Lq, Lk) @ (B*H, Lv, D) -> (B*H, Lq, D) (Note: Lk == Lv assumed for self-attn)
        attn_output = torch.bmm(attn_weights_dropped, v) # [B*H, Lq, head_dim]

        # --- 9. Reshape and Final Projection ---
        # Reshape: [B*H, Lq, D] -> [B, H, Lq, D] -> [B, Lq, H, D] -> [B, Lq, E]
        attn_output = attn_output.view(batch_size, self.num_heads, seq_len_q, self.head_dim).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        attn_output = self.out_proj(attn_output) # [B, Lq, E]

        # --- Format Weights for Output ---
        if need_weights:
             # attn_weights shape: [B*H, Lq, Lk]
             attn_weights = attn_weights.view(batch_size, self.num_heads, seq_len_q, seq_len_k)
             if average_attn_weights:
                 attn_weights = attn_weights.mean(dim=1) # Average across heads -> [B, Lq, Lk]
             # else: attn_weights remains [B, H, Lq, Lk]
        else:
             attn_weights = None

        # --- Handle batch_first = False case ---
        if not self.batch_first:
             attn_output = attn_output.transpose(0, 1) # [Lq, B, E]
             # Note: attn_weights shape doesn't usually depend on batch_first convention

        return attn_output, attn_weights


# (Uses the above custom attention layer)
class DistanceAwareTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first=True, layer_norm_eps=1e-5, activation=F.relu):
        super(DistanceAwareTransformerEncoderLayer, self).__init__()

        # Use the REVISED distance-aware multi-head attention
        self.self_attn = DistanceAwareMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first
            # Add other MHA params if needed (bias, kdim, vdim etc.)
        )

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout for connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation
        self.activation = activation

    def forward(self, src, distances=None, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Input tensor [batch_size, seq_len, d_model]
            distances: Distance matrix [batch_size, seq_len, seq_len]
        """
        # --- Self-Attention Block ---
        # LayerNorm first (common practice, though original was post-LN)
        # src_norm = self.norm1(src) # Pre-LN variation
        attn_output, _ = self.self_attn( # Use REVISED attention
            query=src, # Use src_norm for Pre-LN
            key=src,   # Use src_norm for Pre-LN
            value=src, # Use src_norm for Pre-LN
            distances=distances,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False # Don't need weights from MHA usually
        )
        src = src + self.dropout1(attn_output) # Add residual connection
        src = self.norm1(src) # Post-LN variation (as in original code)

        # --- Feed-Forward Block ---
        # src_norm = self.norm2(src) # Pre-LN variation
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src)))) # Use src_norm for Pre-LN
        src = src + self.dropout2(ff_output) # Add residual connection
        src = self.norm2(src) # Post-LN variation

        return src


# (Fixed weight sharing issue in __init__)
class DistanceAwareTransformerEncoder(nn.Module):
    """
    TransformerEncoder with distance-aware attention - Corrected Init
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers, batch_first=True, layer_norm_eps=1e-5, activation=F.relu):
        super(DistanceAwareTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            DistanceAwareTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=batch_first,
                layer_norm_eps=layer_norm_eps,
                activation=activation
            ) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if num_layers > 0 else None # Final LayerNorm

    def forward(self, src, distances=None, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                distances=distances,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


# (Passes parameters, uses LayerNorm option)
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, dropout_rate, use_batch_norm=False): # Added flag
        super(TransformerEncoder, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        # Option to use BatchNorm or LayerNorm after projection
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
             # Be cautious with BN on sequences, might need permute/reshape logic
             self.input_norm = nn.BatchNorm1d(d_model)
        else:
             self.input_norm = nn.LayerNorm(d_model) # LayerNorm often safer

        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout_rate)

        self.transformer_encoder = DistanceAwareTransformerEncoder(
             d_model=d_model,
             nhead=nhead,
             dim_feedforward=dim_feedforward,
             dropout=dropout_rate,
             num_layers=num_layers,
             batch_first=True # Ensure consistency
             # Pass other relevant params like layer_norm_eps, activation if needed
        )
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)

    def forward(self, src, distances=None, src_mask=None, src_key_padding_mask=None):
        # src shape: [batch_size, seq_len, input_size]
        # distances shape: [batch_size, seq_len, seq_len] or None
        batch_size, seq_len, _ = src.shape

        src = self.input_projection(src) # [B, L, D]

        # Apply normalization
        if self.use_batch_norm:
             # BN expects [N, C] or [N, C, L]. Here C=d_model.
             if seq_len > 1:
                  src_permuted = src.permute(0, 2, 1) # [B, D, L]
                  src_norm = self.input_norm(src_permuted)
                  src = src_norm.permute(0, 2, 1) # [B, L, D]
             # else: Skip BN for L=1? Or handle differently.
        else:
             # LayerNorm applied on the last dimension (d_model)
             src = self.input_norm(src)

        src = self.positional_encoding(src)

        output = self.transformer_encoder(
            src,
            distances=distances,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        return output


class TransformerDecoder(nn.Module):
     # ... (Keep original implementation, maybe switch BN to LN in output) ...
     def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout_rate, num_classes):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.embedding = nn.Linear(num_classes, d_model) # Using Linear for continuous input/output
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout_rate)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout_rate, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Using LayerNorm in output projection might be more stable
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model), # Switched from BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, num_classes)
        )
        self.init_weights()

     def init_weights(self):
        nn.init.kaiming_uniform_(self.embedding.weight)
        if self.embedding.bias is not None: nn.init.zeros_(self.embedding.bias)
        for layer in self.output_projection:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None: nn.init.zeros_(layer.bias)

     # --- [ Keep existing forward, forward_step, generate_square_subsequent_mask methods here ] ---
     # Make sure forward_step aligns or is removed if forward loop is sufficient.
     # The forward loop implementation needs careful checking for norm layers.
     def forward(self, encoder_output, tgt=None, tgt_length=None,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                teacher_forcing_ratio=0.5):
        batch_size = encoder_output.size(0)
        device = encoder_output.device

        if tgt_length is None:
            tgt_length = tgt.size(1) if tgt is not None else encoder_output.size(1)

        # Ensure tgt_length > 0 before creating mask
        if tgt_mask is None and tgt_length > 0:
            tgt_mask = self.generate_square_subsequent_mask(tgt_length).to(device)
        elif tgt_length == 0:
             tgt_mask = None # Or handle error/empty output

        # Start token representation (zeros for Linear embedding)
        decoder_input = torch.zeros(batch_size, 1, self.num_classes).to(device)

        outputs = []

        for t in range(tgt_length):
            current_input_embedded = self.embedding(decoder_input) # [B, current_L, D]
            current_input_embedded = self.positional_encoding(current_input_embedded)

            step_tgt_mask = tgt_mask[:t+1, :t+1] if tgt_mask is not None else None
            # Handle potential padding mask slicing for current length
            step_tgt_key_padding_mask = tgt_key_padding_mask[:, :t+1] if tgt_key_padding_mask is not None else None

            # Decoder forward pass
            output_step = self.transformer_decoder(
                current_input_embedded, encoder_output,
                tgt_mask=step_tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=step_tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            ) # [B, current_L, D]

            # Project the *last* time step's output from the decoder sequence
            last_step_output = output_step[:, -1, :] # [B, D]

            # Apply projection layer by layer
            projected_output = last_step_output
            for layer in self.output_projection:
                 projected_output = layer(projected_output) # LayerNorm works directly

            current_output = projected_output.unsqueeze(1) # [B, 1, num_classes]
            outputs.append(current_output)

            # Determine next input
            use_teacher_force = torch.rand(1).item() < teacher_forcing_ratio if self.training and tgt is not None else False

            if use_teacher_force and t < tgt.size(1) - 1 :
                 next_input_token = tgt[:, t+1:t+2, :] # Ground truth
            else:
                 next_input_token = current_output # Use prediction

            decoder_input = torch.cat([decoder_input, next_input_token], dim=1)

        if not outputs: # Handle edge case of tgt_length=0
            return torch.empty(batch_size, 0, self.num_classes, device=device)

        return torch.cat(outputs, dim=1)

     def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device=self.embedding.weight.device)) == 1).transpose(0, 1) # Ensure mask on correct device
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class Transformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout_rate, num_classes, use_batch_norm_encoder=False):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            input_size=input_size, d_model=d_model, nhead=nhead,
            num_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate, use_batch_norm=use_batch_norm_encoder
        )

        self.encoder_only = (num_decoder_layers == 0)

        if not self.encoder_only:
            self.decoder = TransformerDecoder(
                d_model=d_model, nhead=nhead, num_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward, dropout_rate=dropout_rate, num_classes=num_classes
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model, num_classes)
            )
            self.init_classifier_weights()

    def init_classifier_weights(self):
         if hasattr(self, 'classifier'):
             for layer in self.classifier:
                 if isinstance(layer, nn.Linear):
                     nn.init.kaiming_uniform_(layer.weight)
                     if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, src, distances=None, tgt=None, tgt_length=None, teacher_forcing_ratio=0.5):
        """
        Args:
            src: Source sequence [batch_size, seq_len, input_size]
            distances: Distance matrix [batch_size, seq_len, seq_len] or None
            tgt: Target sequence for teacher forcing (decoder) [batch_size, tgt_seq_len, num_classes]
            ... other args
        """
        src_mask = None
        src_key_padding_mask = None

        # Pass distances to the encoder
        memory = self.encoder(src, distances=distances,
                              src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        if self.encoder_only:
            # Apply classifier head
            batch_size, seq_len, d_model = memory.shape
            memory_flat = memory.reshape(-1, d_model)

            # Apply classifier layer by layer (LayerNorm handles shapes easily)
            output_flat = memory_flat
            for layer in self.classifier:
                 output_flat = layer(output_flat)

            outputs = output_flat.reshape(batch_size, seq_len, -1)
            return outputs
        else:
            # Standard decoder pass (decoder itself isn't distance-aware here)
            outputs = self.decoder(
                memory, tgt, tgt_length,
                teacher_forcing_ratio=teacher_forcing_ratio
                # Pass memory masks if needed
            )
            return outputs

