import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(1)
        hidden = hidden.expand(-1, encoder_outputs.size(1), -1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class LSTM_Attention_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, output_dim=1, dropout_rate=0.3):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_to_hidden = nn.Linear(output_dim, hidden_dim)
    
    def forward(self, x):
        if x.dim() == 2:  # [batch, time_steps]
            x = x.unsqueeze(-1)  # → [batch, time_steps, 1]

        encoder_outputs, (hidden, cell) = self.encoder(x)

        hidden = hidden.permute(1, 0, 2)
        cell = cell.permute(1, 0, 2)
        
        # Chuẩn bị đầu vào cho decoder
        decoder_input = torch.zeros(x.size(0), 1, hidden.size(-1)).to(x.device)
        outputs = []
        
        for t in range(x.size(1)):  # Duyệt qua từng bước thời gian
            decoder_output, (hidden, cell) = self.decoder(
                decoder_input, 
                (hidden.permute(1, 0, 2), cell.permute(1, 0, 2))
            )
            # attention
            last_hidden = hidden.permute(1, 0, 2)[:, -1, :] 
            attn_weights = self.attention(last_hidden, encoder_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

            hidden = hidden.permute(1, 0, 2)
            cell = cell.permute(1, 0, 2)

            # predict
            output = self.fc(context)
            outputs.append(output)

            # update
            decoder_input = self.output_to_hidden(output).unsqueeze(1)
        
        return torch.stack(outputs, dim=1)
    