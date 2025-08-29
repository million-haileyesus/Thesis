import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, encoder_norm_layer):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization
        self.input_bn = encoder_norm_layer
        
        # GRU for encoding input sequence
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate
        )
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.kaiming_uniform_(param)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        batch_size = x.size(0)
        
        # Apply batch normalization
        # Reshape for batch norm
        x_reshaped = x.reshape(-1, x.size(2))
        x_normalized = self.input_bn(x_reshaped)
        x = x_normalized.reshape(batch_size, -1, x.size(2))
        
        # Encoder GRU
        # outputs shape: [batch_size, seq_len, hidden_size]
        # hidden shape: [num_layers, batch_size, hidden_size]
        # CHANGE: Use self.gru and unpack only one state (hidden)
        outputs, hidden = self.gru(x)
        
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout_rate, num_classes, decoder_norm_layer):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # GRU for decoding
        self.gru = nn.GRU(
            input_size=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Hidden state normalization
        self.hidden_bn = decoder_norm_layer
        
        # Output projection remains the same
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            decoder_norm_layer,
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.init_weights()
        
    def init_weights(self):
        # CHANGE: Adjust initialization for GRU parameters
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.kaiming_uniform_(param)
                
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, encoder_outputs, encoder_hidden, target=None, target_length=None, teacher_forcing_ratio=0.5):
        """
        Full sequence decoding
        
        Args:
            encoder_outputs: Outputs from encoder [batch_size, input_seq_len, hidden_size]
            encoder_hidden: Final hidden state from encoder
            target: Target sequence for teacher forcing [batch_size, target_seq_len, num_classes]
            target_length: Length of output sequence to generate
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Sequence of predictions [batch_size, target_seq_len, num_classes]
        """
        batch_size = encoder_outputs.size(0)
        
        # Determine output sequence length
        if target_length is None:
            target_length = target.size(1) if target else encoder_outputs.size(1)
        
        # Initialize first decoder input as zeros (start token)
        decoder_input = torch.zeros(batch_size, 1, self.num_classes).to(encoder_outputs.device)
        
        # Use encoder final hidden state for decoder initialization
        # CHANGE: For GRU, encoder_hidden is the only state
        decoder_hidden = encoder_hidden
        
        # Store outputs
        outputs = []
        
        # Generate output sequence
        for t in range(target_length):
            # Forward pass through decoder for single step
            output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            outputs.append(output)
            
            # Teacher forcing: use ground truth as next input with probability teacher_forcing_ratio
            if target is not None and t < target_length-1 and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = output
        
        # Concatenate outputs along sequence dimension
        return torch.cat(outputs, dim=1)

    def forward_step(self, input_, hidden):
        """
        Process a single decoding step
        
        Args:
            input_: Input tensor for this step [batch_size, 1, output_size]
            hidden: Hidden state from previous step or encoder
            
        Returns:
            output: Prediction for this step [batch_size, 1, output_size]
            hidden: Updated hidden state for next step
        """
        batch_size = input_.size(0)
        
        # Run single GRU step
        # CHANGE: Use self.gru and note it returns only hidden state
        gru_out, hidden = self.gru(input_, hidden)
        # gru_out shape: [batch_size, 1, hidden_size]
        
        # Apply batch norm to hidden state
        gru_out_flat = gru_out.reshape(-1, self.hidden_size)
        gru_out_norm = self.hidden_bn(gru_out_flat)
        gru_out = gru_out_norm.reshape(batch_size, 1, self.hidden_size)
        
        # Apply output layers
        gru_out_flat = gru_out.reshape(-1, self.hidden_size)
        fc_out = self.fc[0](gru_out_flat)  # Linear
        fc_out = self.fc[1](fc_out)         # BatchNorm
        fc_out = self.fc[2](fc_out)         # ReLU
        fc_out = self.fc[3](fc_out)         # Dropout
        output_flat = self.fc[4](fc_out)    # Final linear
        
        # Reshape back
        output = output_flat.reshape(batch_size, 1, self.num_classes)
        
        return output, hidden

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, num_classes, decoder_num_layers=None, use_batch_norm=True):
        super(GRU, self).__init__()

        self.num_classes = num_classes
        # If decoder_num_layers is not specified, use the same as encoder
        if decoder_num_layers is None:
            decoder_num_layers = num_layers

        if use_batch_norm:
            encoder_norm_layer = nn.BatchNorm1d(input_size) 
            decoder_norm_layer = nn.BatchNorm1d(hidden_size) 
        else:
            encoder_norm_layer = nn.LayerNorm(input_size)
            decoder_norm_layer = nn.LayerNorm(hidden_size)
        
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout_rate, encoder_norm_layer)
        
        # Flag to determine if we're using encoder-only architecture
        self.encoder_only = (decoder_num_layers == 0)
        
        # Create decoder only if needed
        if not self.encoder_only:
            self.decoder = Decoder(hidden_size, decoder_num_layers, dropout_rate, num_classes, decoder_norm_layer)
        
        # For encoder-only model, add a classification head
        if self.encoder_only:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                decoder_norm_layer,
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, num_classes)
            )
            # Initialize weights for classifier
            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
    def forward(self, src, target=None, target_length=None, teacher_forcing_ratio=0.5):
        """
        Full sequence to sequence model
        
        Args:
            src: Source sequence [batch_size, src_seq_len, input_size]
            target: Target sequence for teacher forcing [batch_size, target_seq_len, num_classes]
            target_length: Length of output sequence to generate
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Sequence of predictions [batch_size, target_seq_len, num_classes] 
                     or [batch_size, src_seq_len, num_classes] for encoder-only
        """
        # Encode input sequence
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        # For encoder-only architecture, apply classification head directly
        if self.encoder_only:
            batch_size, seq_len, hidden_size = encoder_outputs.shape
            
            # Apply classifier to each position in the sequence
            encoder_outputs_flat = encoder_outputs.reshape(-1, hidden_size)
            output_flat = self.classifier[0](encoder_outputs_flat)  # Linear
            output_flat = self.classifier[1](output_flat)           # Norm
            output_flat = self.classifier[2](output_flat)           # ReLU
            output_flat = self.classifier[3](output_flat)           # Dropout
            output_flat = self.classifier[4](output_flat)           # Final linear
            
            # Reshape back to sequence format
            outputs = output_flat.reshape(batch_size, seq_len, -1)
            return outputs
        
        # For encoder-decoder architecture, use the decoder
        else:
            outputs = self.decoder(
                encoder_outputs,
                encoder_hidden,  # CHANGE: Pass only the hidden state (no cell state)
                target,
                target_length,
                teacher_forcing_ratio
            )
            
            return outputs
