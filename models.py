import torch
import torch.nn as nn
import torch.nn.functional as F
from performer_pytorch import SelfAttention
from linformer import LinformerSelfAttention
from scalable_softmax import ScalableSoftmax
from entmax import sparsemax, entmax15

def adassmax(scores, dim=1, alpha_factor=1.0):
    n = scores.size(dim)
    scaled_scores = scores * torch.rsqrt(torch.tensor(n, dtype=scores.dtype, device=scores.device))
    temp_scores = scaled_scores * alpha_factor
    
    max_scores = torch.max(temp_scores, dim=dim, keepdim=True).values
    stabilized_scores = temp_scores - max_scores
    
    exp_scores = torch.exp(stabilized_scores)
    sum_exp = torch.sum(exp_scores, dim=dim, keepdim=True)
    weights = exp_scores / (sum_exp + 1e-8)
    
    return weights

class ImportanceScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

class LinearAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, attention_type='adassmax', alpha_factor=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.attention_type = attention_type

        if self.attention_type not in ['performer', 'linformer']:
            self.importance_scorer = ImportanceScorer(embed_dim)
        else:
            self.importance_scorer = None

        if self.attention_type == 'adassmax':
            self.log_alpha_factor = nn.Parameter(torch.log(torch.tensor(float(alpha_factor))))
        else:
            self.register_parameter('log_alpha_factor', None)

        if self.attention_type == 'ssmax':
            self.ssmax_layer = ScalableSoftmax()
        elif self.attention_type == 'performer':
            self.performer_layer = SelfAttention(dim=embed_dim, heads=8, causal=False)
        elif self.attention_type == 'linformer':
            k_dim = min(self.input_dim, 256)
            self.linformer_layer = LinformerSelfAttention(
                dim=embed_dim,
                seq_len=self.input_dim,
                heads=8,
                k=k_dim
            )
        else:
            self.ssmax_layer = None

    def forward(self, x):
        if self.attention_type == 'performer':
            attended_x = self.performer_layer(x)
            batch_size, num_features, _ = x.shape
            scores = torch.zeros(batch_size, num_features, device=x.device)
            weights = torch.zeros(batch_size, num_features, 1, device=x.device)
            return attended_x, scores, weights

        if self.attention_type == 'linformer':
            attended_x = self.linformer_layer(x)
            batch_size, num_features, _ = x.shape
            scores = torch.zeros(batch_size, num_features, device=x.device)
            weights = torch.zeros(batch_size, num_features, 1, device=x.device)
            return attended_x, scores, weights

        scores = self.importance_scorer(x)

        if self.attention_type == 'adassmax':
            alpha = F.softplus(self.log_alpha_factor) + 1
            alpha = torch.clamp(alpha, min=1.0, max=10.0)
            weights = adassmax(scores, dim=1, alpha_factor=alpha).unsqueeze(-1)
        elif self.attention_type == 'softmax':
            weights = F.softmax(scores, dim=1).unsqueeze(-1)
        elif self.attention_type == 'sparsemax':
            weights = sparsemax(scores, dim=1).unsqueeze(-1)
        elif self.attention_type == 'entmax15':
            weights = entmax15(scores, dim=1).unsqueeze(-1)
        elif self.attention_type == 'ssmax':
            if self.ssmax_layer is None:
                raise ValueError("ssmax attention was selected, but the layer is not initialized.")
            weights = self.ssmax_layer(scores).unsqueeze(-1)
        else: # Handles 'none' case implicitly
            raise ValueError(f"Unknown or unsupported attention type for this block: {self.attention_type}")

        x_weighted = x * weights

        Q_prime = F.elu(x_weighted) + 1
        K_prime = F.elu(x_weighted) + 1

        KV = torch.bmm(K_prime.transpose(1, 2), x_weighted)
        numerator = torch.bmm(Q_prime, KV)
        denominator = (Q_prime * K_prime.sum(dim=1, keepdim=True)).sum(dim=2)
        denominator = denominator.unsqueeze(-1) + 1e-6

        attended_x = numerator / denominator
        return attended_x, scores, weights

class AdaptiveAnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128, p_mask=0.15,
                 dynamic_masking=False, mask_range=(0.1, 0.3),
                 inject_noise=False, noise_std=0.1, attention_type='adassmax',
                 alpha_factor=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.p_mask = p_mask
        self.dynamic_masking = dynamic_masking
        self.mask_range = mask_range
        self.inject_noise = inject_noise
        self.noise_std = noise_std
        self.embedding = nn.Linear(1, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)
        if attention_type != 'none':
            self.linear_attn = LinearAttention(input_dim=self.input_dim, embed_dim=embed_dim, attention_type=attention_type, alpha_factor=alpha_factor)
        else:
            self.linear_attn = None
        self.encoder_block = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2)
        )
        self.encoder_linear = nn.Linear(128, embed_dim)
        self.encoder_skip = nn.Linear(128, embed_dim)
        self.latent_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        x_original = x.clone()
        if self.training:
            curr_p_mask = self.p_mask
            if self.dynamic_masking:
                curr_p_mask = torch.empty(1).uniform_(*self.mask_range).item()
            mask = (torch.rand(x.shape, device=x.device) < curr_p_mask)
            if self.inject_noise:
                noise = torch.randn_like(x) * self.noise_std
                x_masked = torch.where(mask, noise, x)
            else:
                x_masked = x.masked_fill(mask, 0)
        else:
            mask = None
            x_masked = x
        x_masked = x_masked.unsqueeze(-1)
        x_emb = self.embedding(x_masked)
        x_emb_normalized = self.embed_norm(x_emb)

        if self.linear_attn:
            x_attn, scores, attn_weights = self.linear_attn(x_emb_normalized)
            x_pooled = x_attn.mean(dim=1)
        else:
            x_pooled = x_emb_normalized.mean(dim=1)
            batch_size, num_features = x.shape[0], x.shape[1]
            scores = torch.zeros(batch_size, num_features, device=x.device)
            attn_weights = torch.zeros(batch_size, num_features, 1, device=x.device)

        encoded = self.encoder_block(x_pooled)
        latent_combined = self.encoder_linear(encoded) + self.encoder_skip(encoded)
        latent = self.latent_norm(latent_combined)
        x_reconstructed = self.decoder(latent)
        rec_error = torch.mean((x_reconstructed - x_original)**2, dim=1)
        cosine_sim = F.cosine_similarity(x_reconstructed, x_original, dim=1)
        cosine_dissim = 1 - cosine_sim
        rec_error = rec_error.unsqueeze(1)
        cosine_dissim = cosine_dissim.unsqueeze(1)
        composite_feature = torch.cat([latent, rec_error, cosine_dissim], dim=1)
        if self.training:
            return x_reconstructed, composite_feature, scores, mask, x_original, attn_weights
        else:
            return x_reconstructed, composite_feature, scores, mask, None, attn_weights
