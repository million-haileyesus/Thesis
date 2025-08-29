import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the token embeddings for the Transformer.
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
        pe = pe.unsqueeze(0) # Shape: [1, max_seq_length, d_model]
        
        # Register as buffer (not a parameter but should be saved and loaded with model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    """
    Custom Transformer Encoder module.
    Projects input, adds positional encoding, and applies Transformer encoder layers.
    """
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, dropout_rate):
        super(TransformerEncoder, self).__init__()
        
        # Input projection to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)
        self.input_ln = nn.LayerNorm(d_model) 
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout_rate)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.kaiming_uniform_(self.input_projection.weight, nonlinearity="relu")
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape: [batch_size, seq_len, input_size]
        projected_src = self.input_projection(src)
        normalized_src = self.input_ln(projected_src)
        positioned_src = self.positional_encoding(normalized_src)
        output = self.transformer_encoder(positioned_src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout_rate, num_classes):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Embedding for decoder input (from num_classes to d_model)
        self.embedding = nn.Linear(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout_rate)
        
        # Transformer decoder layers
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
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, num_classes)
        )
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.kaiming_uniform_(self.embedding.weight, nonlinearity="relu")
        if self.embedding.bias is not None:
            nn.init.zeros_(self.embedding.bias)
        
        for layer in self.output_projection:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def generate_square_subsequent_mask(self, sz, device):
        """Generate a square causal mask for the sequence of size sz."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, encoder_output, tgt=None, tgt_length=None, 
                teacher_forcing_ratio=0.5, memory_key_padding_mask=None):
        """
        Autoregressive decoding for sequence-to-sequence tasks
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Determine output sequence length
        if tgt_length is None:
            if tgt is not None:
                tgt_length = tgt.size(1)
            else:
                tgt_length = encoder_output.size(1)
        
        # Start-of-sequence token (zeros)
        sos_token = torch.zeros(batch_size, 1, self.num_classes, device=device)
        
        if tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Teacher forcing: use ground truth as input
            # Prepend SOS token to target sequence
            decoder_input = torch.cat([sos_token, tgt[:, :-1, :]], dim=1)
            
            # Create causal mask
            seq_len = decoder_input.size(1)
            tgt_mask = self.generate_square_subsequent_mask(seq_len, device)
            
            # Forward pass
            embedded_input = self.embedding(decoder_input)
            positioned_input = self.positional_encoding(embedded_input)
            
            decoder_output = self.transformer_decoder(
                positioned_input, 
                encoder_output,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # Handle 3D tensor reshaping if needed
            batch_size, seq_len, d_model = decoder_output.shape
            decoder_output_flat = decoder_output.reshape(-1, d_model)
            output_flat = self.output_projection(decoder_output_flat)
            output = output_flat.reshape(batch_size, seq_len, self.num_classes)
            return output
        else:
            # Autoregressive generation
            outputs = []
            decoder_input = sos_token
            
            for t in range(tgt_length):
                seq_len = decoder_input.size(1)
                tgt_mask = self.generate_square_subsequent_mask(seq_len, device)
                
                embedded_input = self.embedding(decoder_input)
                positioned_input = self.positional_encoding(embedded_input)
                
                decoder_output = self.transformer_decoder(
                    positioned_input, 
                    encoder_output,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
                
                # Get last timestep output and apply projection
                last_output = decoder_output[:, -1, :]  # [batch_size, d_model]
                step_output = self.output_projection(last_output).unsqueeze(1)  # [batch_size, 1, num_classes]
                outputs.append(step_output)
                
                # Use predicted output as next input
                next_input = F.softmax(step_output, dim=-1)
                decoder_input = torch.cat([decoder_input, next_input], dim=1)
            
            return torch.cat(outputs, dim=1)

class Transformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, dropout_rate, num_classes, max_seq_length=5000):
        super(Transformer, self).__init__()

        if num_encoder_layers == 0 and num_decoder_layers == 0:
            raise ValueError("Both num_encoder_layers and num_decoder_layers cannot be zero.")

        self.d_model = d_model
        self.num_classes = num_classes
        self.encoder_only = (num_decoder_layers == 0) and (num_encoder_layers > 0)
        self.decoder_only = (num_encoder_layers == 0) and (num_decoder_layers > 0)
        self.encoder_decoder = (num_encoder_layers > 0) and (num_decoder_layers > 0)

        # Create encoder if needed
        if self.encoder_only or self.encoder_decoder:
            self.encoder = TransformerEncoder(
                input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout_rate
            )
        
        # Create decoder if needed
        if self.decoder_only or self.encoder_decoder:
            self.decoder = TransformerDecoder(
                d_model, nhead, num_decoder_layers, dim_feedforward, dropout_rate, num_classes
            )

        # For encoder-only model (sequence classification/tagging)
        if self.encoder_only:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model, num_classes)
            )
            self._init_classifier_weights()

        # For decoder-only model
        if self.decoder_only:
            self.input_embedding = nn.Linear(input_size, d_model)
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout_rate)
            
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout_rate, batch_first=True
            )
            self.decoder_stack = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)
            self.output_projection = nn.Linear(d_model, num_classes)
            self._init_decoder_only_weights()

    def _init_classifier_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _init_decoder_only_weights(self):
        nn.init.kaiming_uniform_(self.input_embedding.weight, nonlinearity="relu")
        if self.input_embedding.bias is not None:
            nn.init.zeros_(self.input_embedding.bias)
        nn.init.kaiming_uniform_(self.output_projection.weight, nonlinearity="relu")
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def generate_causal_mask(self, sz, device):
        """Generates a square causal mask for a sequence of size sz."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src, tgt=None, tgt_length=None, teacher_forcing_ratio=0.5,
                src_key_padding_mask=None):
        """
        Forward pass for different transformer architectures
        
        Args:
            src: Source sequence [batch_size, seq_len, input_size]
            tgt: Target sequence for teacher forcing [batch_size, tgt_seq_len, num_classes]
            tgt_length: Desired output length for generation
            teacher_forcing_ratio: Probability of using teacher forcing
            src_key_padding_mask: Padding mask for source sequence
        """
        
        if self.encoder_only:
            # Encoder-only: sequence classification/tagging
            encoded = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
            output = self.classifier(encoded)
            return output
        
        elif self.encoder_decoder:
            # Encoder-decoder: sequence-to-sequence
            encoded = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
            output = self.decoder(
                encoded, 
                tgt=tgt, 
                tgt_length=tgt_length,
                teacher_forcing_ratio=teacher_forcing_ratio,
                memory_key_padding_mask=src_key_padding_mask
            )
            return output

        elif self.decoder_only:
            # Decoder-only: causal language modeling
            batch_size, seq_len, _ = src.shape
            device = src.device
            
            embedded = self.input_embedding(src)
            positioned = self.positional_encoding(embedded)
            
            causal_mask = self.generate_causal_mask(seq_len, device)
            
            decoded = self.decoder_stack(
                positioned, 
                mask=causal_mask, 
                src_key_padding_mask=src_key_padding_mask
            )
            
            output = self.output_projection(decoded)
            return output
        
        else:
            raise RuntimeError("Model configuration is not recognized.")

