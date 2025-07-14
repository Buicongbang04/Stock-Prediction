import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_length, in_dim, embed_dim):
        super().__init__()
        self.patch_length = patch_length
        self.proj = nn.Linear(patch_length * in_dim, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, in_dim)
        B, L, D = x.shape
        # cắt thành các patch và project lên embed_dim
        n_patch = L // self.patch_length
        x = x[:, :n_patch*self.patch_length, :] # bỏ thừa
        x = x.reshape(B, n_patch, self.patch_length*D)
        x = self.proj(x) # (batch, n_patch, embed_dim)
        return x

class VariateEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.proj = nn.Linear(seq_len, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len)
        # Đầu vào từng biến ngoại sinh, project thành token
        x = self.proj(x) # (batch, embed_dim)
        return x

class TimeXerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, exo_token_num):
        super().__init__()
        # Self-attention cho endogenous patch tokens + global token
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Cross-attention: global token với exogenous variate tokens
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Feed Forward
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, p_en, g_en, v_ex):
        # Patch-to-Patch + Global (self-attention)
        tokens = torch.cat([p_en, g_en.unsqueeze(1)], dim=1) # (B, N_patch+1, D)
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.layernorm(tokens + attn_out)

        # tách lại patch tokens + global token
        p_en_new = tokens[:, :-1, :]
        g_en_new = tokens[:, -1, :] # (B, D)

        # Cross-attention: global token Q, exogenous tokens K,V
        g_en_input = g_en_new.unsqueeze(1)
        attn_out, _ = self.cross_attn(g_en_input, v_ex, v_ex)
        g_en_new = self.layernorm(g_en_new + attn_out.squeeze(1))

        # FFN
        p_en_new = self.ffn(p_en_new)
        g_en_new = self.ffn(g_en_new)

        return p_en_new, g_en_new

class TimeXer(nn.Module):
    def __init__(self, 
                in_dim,                 # đầu vào biến endo (1 hoặc nhiều)
                exo_num,                # số biến exogenous
                exo_seq_len,            # chiều dài chuỗi exo
                patch_length,
                embed_dim,
                num_layers,
                num_heads,
                pred_len):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_length, in_dim, embed_dim)
        self.global_token = nn.Parameter(torch.randn(embed_dim)) # learnable token
        self.variate_embed = nn.ModuleList([VariateEmbedding(exo_seq_len, embed_dim) for _ in range(exo_num)])
        self.blocks = nn.ModuleList([TimeXerBlock(embed_dim, num_heads, exo_num) for _ in range(num_layers)])
        self.head = nn.Linear(embed_dim, pred_len) # dự báo pred_len time-step

    def forward(self, x_endo, x_exo):
        """
        x_endo: (batch, seq_len, in_dim)
        x_exo:  list of [ (batch, exo_seq_len), ... ]  -- exo_num biến
        """
        patch_tokens = self.patch_embed(x_endo)                   # (B, N_patch, D)
        B = x_endo.size(0)
        global_token = self.global_token.unsqueeze(0).expand(B, -1) # (B, D)

        # tạo exo tokens
        v_ex = [emb(x_exo[i]) for i, emb in enumerate(self.variate_embed)] # List[(B, D)]
        v_ex = torch.stack(v_ex, dim=1) # (B, exo_num, D)

        # Lặp qua các block
        p, g = patch_tokens, global_token
        for block in self.blocks:
            p, g = block(p, g, v_ex)

        # Kết hợp Patch tokens + Global token để dự báo
        out = torch.cat([p.mean(dim=1), g], dim=-1) # đơn giản hóa, hoặc chỉ g
        pred = self.head(g) # đơn giản: chỉ dùng global token
        return pred