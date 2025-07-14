import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, kernel_size=3):
        super(CNNEmbedding, self).__init__()
        # 1D convolution (sử dụng padding circular)
        self.conv1d = nn.Conv1d(in_channels=input_dim,
                                out_channels=embed_dim,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=kernel_size // 2,
                                padding_mode='circular')

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (batch, seq_len, embed_dim)
        return x

class StockformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(StockformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, embed_dim*4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim*4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, src):
        # src: (batch, seq_len, embed_dim)
        attn_out, _ = self.self_attn(src, src, src)
        src = self.norm1(src + attn_out)

        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + ff)

        # Down-sample (Self-attention distilling): reduce seq_len by half
        src = src.transpose(1,2)
        src = self.pool(src)
        src = src.transpose(1,2)
        return src

class Stockformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, out_dim=1):
        super(Stockformer, self).__init__()
        # Embedding Layer
        self.embedding = CNNEmbedding(input_dim, embed_dim)
        # Encoder Layers (stacked)
        self.encoders = nn.ModuleList([
            StockformerEncoderLayer(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        # Output Layer
        self.fc_out = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.embedding(x)
        for encoder in self.encoders:
            x = encoder(x)
        # Lấy ra timestep cuối cùng hoặc dùng pool
        out = self.fc_out(x[:, -1, :])
        return out