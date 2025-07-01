import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        # decoder_hidden: [batch_size, hidden_dim]
        seq_len = encoder_outputs.size(1)

        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((encoder_outputs, decoder_hidden), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)

        attention_weights = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attention_weights, dim=1)


class LSTM_Attention_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.3):
        super(LSTM_Attention_LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Attention
        self.attention = Attention(hidden_dim)

        # Decoder
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        encoder_outputs, (hidden, cell) = self.encoder(x)
        decoder_input = encoder_outputs[:, -1:, :]  # last output as initial input for decoder

        all_outputs = []
        seq_len = x.size(1)

        for _ in range(seq_len):
            # Calculate attention
            attn_weights = self.attention(encoder_outputs, hidden[-1])
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

            # Decode
            decoder_output, (hidden, cell) = self.decoder(attn_applied, (hidden, cell))
            output = self.fc_out(decoder_output)
            all_outputs.append(output)

        outputs = torch.cat(all_outputs, dim=1)
        return outputs

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
        return outputs
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)