import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the token embeddings for the Transformer
    """
    def __init__(self, d_model, max_seq_length=5000, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter but should be saved and loaded with model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class DistanceAwareMultiheadAttention(nn.MultiheadAttention):
    """
    Extends MultiheadAttention to incorporate distance-based attention bias
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super(DistanceAwareMultiheadAttention, self).__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        
        # Add a learnable scaling factor for distance influence
        self.distance_scaling = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, query, key, value, distances=None, 
                key_padding_mask=None, need_weights=True, 
                attn_mask=None, average_attn_weights=True):
        """
        Forward pass with optional distance-based attention bias
        
        Args:
            query, key, value: Input tensors
            distances: [batch_size, seq_len, seq_len] tensor of distances between elements
                      (smaller values = closer to ball, higher attention)
            other args: Standard MultiheadAttention arguments
        """
        # Call parent forward method to get attention output and weights
        if hasattr(self, '_scaled_dot_product_attention') and callable(self._scaled_dot_product_attention):
            # For PyTorch versions that support this
            attn_output, attn_weights = super().forward(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights
            )
            
            # Apply distance-based bias if provided
            if distances is not None and need_weights:
                # Convert distances to attention bias (smaller distances → higher attention)
                # Normalize distances to [0, 1] range (inverted so smaller distances get higher values)
                distance_bias = 1.0 - (distances / torch.max(distances, dim=-1, keepdim=True)[0])
                
                # Scale with learnable parameter and apply as multiplicative bias
                distance_bias = torch.exp(self.distance_scaling * distance_bias)
                
                # Apply to attention weights
                attn_weights = attn_weights * distance_bias
                attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)  # Re-normalize
                
                # Recompute attention output with new weights
                attn_output = torch.bmm(attn_weights, value)
            
            return attn_output, attn_weights if need_weights else None
        else:
            # Fallback for older PyTorch versions
            attn_output, attn_weights = super().forward(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask
            )
            
            # Apply distance-based bias if provided (same logic as above)
            if distances is not None and need_weights:
                distance_bias = 1.0 - (distances / torch.max(distances, dim=-1, keepdim=True)[0])
                distance_bias = torch.exp(self.distance_scaling * distance_bias)
                attn_weights = attn_weights * distance_bias
                attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
                attn_output = torch.bmm(attn_weights, value)
            
            return attn_output, attn_weights if need_weights else None

class DistanceAwareTransformerEncoderLayer(nn.Module):
    """
    TransformerEncoderLayer with distance-aware attention
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first=True):
        super(DistanceAwareTransformerEncoderLayer, self).__init__()
        
        # Distance-aware multi-head attention
        self.self_attn = DistanceAwareMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = F.relu
    
    def forward(self, src, distances=None, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass with distance-aware attention
        
        Args:
            src: Input tensor [batch_size, seq_len, d_model]
            distances: Distance matrix [batch_size, seq_len, seq_len]
            src_mask: Optional mask for src sequence
            src_key_padding_mask: Optional padding mask
        """
        # Self-attention block with distance awareness
        src2, _ = self.self_attn(
            query=src,
            key=src,
            value=src,
            distances=distances,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class DistanceAwareTransformerEncoder(nn.Module):
    """
    TransformerEncoder with distance-aware attention
    """
    def __init__(self, encoder_layers, num_layers):
        super(DistanceAwareTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layers for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, src, distances=None, mask=None, src_key_padding_mask=None):
        output = src
        
        for layer in self.layers:
            output = layer(
                output,
                distances=distances,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask
            )
        
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, dropout_rate):
        super(TransformerEncoder, self).__init__()
        
        # Input projection to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)
        self.input_bn = nn.BatchNorm1d(d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout_rate)
        
        # Create distance-aware encoder layer
        encoder_layer = DistanceAwareTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Create distance-aware encoder
        self.transformer_encoder = DistanceAwareTransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        self.init_weights()
        
    def init_weights(self):
        # Initialize weights
        nn.init.kaiming_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
    
    def forward(self, src, distances=None, src_mask=None, src_key_padding_mask=None):
        # src shape: [batch_size, seq_len, input_size]
        # distances shape: [batch_size, seq_len, seq_len] or None
        batch_size = src.size(0)
        
        # Project input to d_model dimensions
        src = self.input_projection(src)
        
        # Apply batch normalization
        src_reshaped = src.reshape(-1, src.size(2))
        src_normalized = self.input_bn(src_reshaped)
        src = src_normalized.reshape(batch_size, -1, src.size(2))
        
        # Add positional encoding
        src = self.positional_encoding(src)
        
        # Apply transformer encoder with distance awareness
        output = self.transformer_encoder(
            src,
            distances=distances,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout_rate, num_classes):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Embedding for decoder input
        self.embedding = nn.Linear(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout_rate)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, num_classes)
        )
        
        self.init_weights()
    
    def init_weights(self):
        # Initialize weights
        nn.init.kaiming_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
        
        for layer in self.output_projection:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, encoder_output, tgt=None, tgt_length=None, 
                tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                teacher_forcing_ratio=0.5):
        """
        Full sequence decoding
        
        Args:
            encoder_output: Output from encoder [batch_size, src_seq_len, d_model]
            tgt: Target sequence for teacher forcing [batch_size, tgt_seq_len, num_classes]
            tgt_length: Length of output sequence to generate
            tgt_mask: Mask for target sequence
            memory_mask: Mask for encoder outputs
            tgt_key_padding_mask: Padding mask for target sequence
            memory_key_padding_mask: Padding mask for encoder outputs
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Sequence of predictions [batch_size, tgt_seq_len, num_classes]
        """
        batch_size = encoder_output.size(0)
        
        # Determine output sequence length
        if tgt_length is None:
            tgt_length = tgt.size(1) if tgt is not None else encoder_output.size(1)
        
        # Create square subsequent mask for target sequence
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt_length).to(encoder_output.device)
        
        # Initialize first decoder input as zeros (start token)
        decoder_input = torch.zeros(batch_size, 1, self.num_classes).to(encoder_output.device)
        
        # Store outputs
        outputs = []
        
        # Generate output sequence
        for t in range(tgt_length):
            # Current decoder input sequence
            current_input = decoder_input
            
            # Forward pass through decoder for current sequence
            output = self.forward_step(current_input, encoder_output, tgt_mask[:t+1, :t+1], memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask)
            # Output shape: [batch_size, t+1, num_classes]
            
            # Extract just the last timestep prediction
            current_output = output[:, -1:, :]
            outputs.append(current_output)
            
            # Teacher forcing: use ground truth as next input with probability teacher_forcing_ratio
            if tgt is not None and t < tgt_length-1 and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = torch.cat([decoder_input, tgt[:, t:t+1, :]], dim=1)
            else:
                decoder_input = torch.cat([decoder_input, current_output], dim=1)
        
        # Concatenate outputs along sequence dimension
        return torch.cat(outputs, dim=1)
    
    def forward_step(self, tgt, memory, tgt_mask=None, memory_mask=None,
                   tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Process target sequence with transformer decoder
        
        Args:
            tgt: Target sequence [batch_size, tgt_seq_len, num_classes]
            memory: Encoder output [batch_size, src_seq_len, d_model]
            tgt_mask: Mask for target sequence
            memory_mask: Mask for encoder outputs
            tgt_key_padding_mask: Padding mask for target sequence
            memory_key_padding_mask: Padding mask for encoder outputs
            
        Returns:
            output: Prediction [batch_size, tgt_seq_len, num_classes]
        """
        batch_size = tgt.size(0)
        
        # Embed target sequence
        tgt_embedded = self.embedding(tgt)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # Apply transformer decoder
        decoder_output = self.transformer_decoder(
            tgt_embedded, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask
        )
        # decoder_output shape: [batch_size, tgt_seq_len, d_model]
        
        # Apply output projection
        decoder_output_flat = decoder_output.reshape(-1, self.d_model)
        output_flat = self.output_projection[0](decoder_output_flat)  # Linear
        output_flat = self.output_projection[1](output_flat)          # BatchNorm
        output_flat = self.output_projection[2](output_flat)          # ReLU
        output_flat = self.output_projection[3](output_flat)          # Dropout
        output_flat = self.output_projection[4](output_flat)          # Final linear
        
        # Reshape back
        output = output_flat.reshape(batch_size, -1, self.num_classes)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class Transformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, dropout_rate, num_classes):
        super(Transformer, self).__init__()
        
        self.encoder = TransformerEncoder(
            input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout_rate
        )
        
        # Flag to determine if we're using encoder-only architecture
        self.encoder_only = (num_decoder_layers == 0)
        
        # Create decoder only if needed
        if not self.encoder_only:
            self.decoder = TransformerDecoder(
                d_model, nhead, num_decoder_layers, dim_feedforward, dropout_rate, num_classes
            )
        
        # For encoder-only model, add a classification head
        if self.encoder_only:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model, num_classes)
            )
            # Initialize weights for classifier
            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, src, distances=None, tgt=None, tgt_length=None, teacher_forcing_ratio=0.5):
        """
        Model forward pass with optional distance-based attention
        
        Args:
            src: Source sequence [batch_size, seq_len, input_size]
            distances: Distance matrix [batch_size, seq_len, seq_len] or None
            tgt: Target sequence for teacher forcing [batch_size, tgt_seq_len, num_classes]
            tgt_length: Length of output sequence to generate
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Sequence of predictions
        """
        # Generate masks (optional, can be set to None to use defaults)
        src_mask = None
        src_key_padding_mask = None
        
        # Encode input sequence with distance awareness
        memory = self.encoder(src, distances, src_mask, src_key_padding_mask)
        
        # For encoder-only architecture, apply classification head directly
        if self.encoder_only:
            batch_size, seq_len, d_model = memory.shape
            
            # Apply classifier to each position in the sequence
            memory_flat = memory.reshape(-1, d_model)
            output_flat = self.classifier[0](memory_flat)  # Linear
            output_flat = self.classifier[1](output_flat)  # BatchNorm
            output_flat = self.classifier[2](output_flat)  # ReLU
            output_flat = self.classifier[3](output_flat)  # Dropout
            output_flat = self.classifier[4](output_flat)  # Final linear
            
            # Reshape back to sequence format
            outputs = output_flat.reshape(batch_size, seq_len, -1)
            return outputs
        
        # For encoder-decoder architecture, use the decoder
        else:
            memory_mask = None
            tgt_key_padding_mask = None
            memory_key_padding_mask = None
            
            # Generate target mask if target sequence is provided
            tgt_mask = None
            if tgt is not None:
                tgt_mask = self.decoder.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
            
            # Decode to generate output sequence
            outputs = self.decoder(
                memory, tgt, tgt_length,
                tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask,
                teacher_forcing_ratio
            )
            
            return outputs
        