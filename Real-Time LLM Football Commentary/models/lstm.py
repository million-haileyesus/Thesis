import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.input_bn = nn.BatchNorm1d(input_size)

        # Define encoder and decoder
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(num_classes, hidden_size, num_layers, batch_first=True)

        self.hidden_bn = nn.BatchNorm1d(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

        self.init_weights()

    def init_weights(self):
        for name, param in self.encoder.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)

        for name, param in self.decoder.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x, target_length=None):
        # Add target_length parameter or use x.size(1) as default
        if target_length is None:
            target_length = x.size(1)  # Use input sequence length as default

        batch_size = x.size(0)

        # x shape: [batch_size, seq_len, features]
        # Transpose to: [batch_size, features, seq_len]
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        # Transpose back: [batch_size, seq_len, features]
        x = x.transpose(1, 2)

        # Encoder
        _, (h_n, c_n) = self.encoder(x)

        # Decoder
        decoder_input = torch.zeros(batch_size, 1, self.num_classes).to(x.device)
        decoder_hidden = (h_n, c_n)
        outputs = []

        for t in range(target_length):
            out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # out shape: [batch_size, 1, hidden_size]
            # Transpose to: [batch_size, hidden_size, 1]
            out = out.transpose(1, 2)
            out = self.hidden_bn(out)
            # Transpose back: [batch_size, 1, hidden_size]
            out = out.transpose(1, 2)

            prediction = self.fc(out.squeeze(1)).unsqueeze(1)
            outputs.append(prediction)
            decoder_input = prediction

        return torch.cat(outputs, dim=1)
