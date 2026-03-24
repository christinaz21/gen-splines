"""
generator.py — Generative Spline Prior Gθ(z) → Control Points

Contains both architectures and auto-detects from checkpoint:
  - PerCurveGenerator: z + curve_embedding → shared MLP → (K, 3) per curve
  - ConvGenerator: z + curve_embedding + pos_enc → coarse MLP → 1D conv refine → (K, 3)

Auto-detection: load_generator() inspects checkpoint keys to pick the right class.
"""

import torch
import torch.nn as nn
import math


# =========================================================================
# V2: Per-Curve MLP Generator (best drift: 0.19)
# =========================================================================

class PerCurveGenerator(nn.Module):
    """Per-curve conditional generation with shared decoder weights."""

    def __init__(self, latent_dim=128, num_curves=40, K=8, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_curves = num_curves
        self.K = K

        self.curve_embeddings = nn.Embedding(num_curves, hidden_dim)
        self.scene_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.curve_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, K * 3),
        )

    def forward(self, z):
        B = z.shape[0]
        N = self.num_curves
        scene_feat = self.scene_proj(z)
        curve_ids = torch.arange(N, device=z.device)
        curve_emb = self.curve_embeddings(curve_ids)
        scene_exp = scene_feat.unsqueeze(1).expand(B, N, -1)
        curve_exp = curve_emb.unsqueeze(0).expand(B, N, -1)
        combined = torch.cat([scene_exp, curve_exp], dim=-1)
        cp_flat = self.curve_decoder(combined)
        return cp_flat.view(B, N, self.K, 3)

    def generate(self, num_samples=1, device="cuda"):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            return self.forward(z)


# =========================================================================
# V3: Conv Refiner Generator
# =========================================================================

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim=64, max_freq=16):
        super().__init__()
        freqs = torch.linspace(1.0, max_freq, dim // 2)
        self.register_buffer("freqs", freqs)

    def forward(self, K, device="cuda"):
        t = torch.linspace(0, 1, K, device=device).unsqueeze(-1)
        args = 2 * math.pi * t * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class CurveRefiner(nn.Module):
    def __init__(self, channels=64, num_layers=4):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = x + layer(x)
        return x


class ConvGenerator(nn.Module):
    """Template + position encoding + 1D conv refinement."""

    def __init__(self, latent_dim=128, num_curves=40, K=8, hidden_dim=256, pos_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_curves = num_curves
        self.K = K

        self.curve_embeddings = nn.Embedding(num_curves, hidden_dim)
        self.pos_enc = SinusoidalPositionEncoding(dim=pos_dim)
        self.scene_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )

        conv_ch = hidden_dim // 4
        self.coarse_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + pos_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, conv_ch),
        )
        self.refiner = CurveRefiner(channels=conv_ch, num_layers=4)
        self.to_coords = nn.Sequential(nn.Linear(conv_ch, 32), nn.GELU(), nn.Linear(32, 3))

    def forward(self, z):
        B = z.shape[0]
        N, K = self.num_curves, self.K
        device = z.device

        scene_feat = self.scene_proj(z)
        curve_emb = self.curve_embeddings(torch.arange(N, device=device))
        pos = self.pos_enc(K, device=device)

        scene_exp = scene_feat[:, None, None, :].expand(B, N, K, -1)
        curve_exp = curve_emb[None, :, None, :].expand(B, N, K, -1)
        pos_exp = pos[None, None, :, :].expand(B, N, K, -1)
        combined = torch.cat([scene_exp, curve_exp, pos_exp], dim=-1)

        coarse = self.coarse_decoder(combined)
        feat = coarse.reshape(B * N, K, -1).permute(0, 2, 1)
        refined = self.refiner(feat).permute(0, 2, 1).reshape(B, N, K, -1)
        return self.to_coords(refined)

    def generate(self, num_samples=1, device="cuda"):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            return self.forward(z)


# =========================================================================
# Auto-detect + load
# =========================================================================

# Alias for backward compatibility
SplineGenerator = PerCurveGenerator


def load_generator(path, device="cuda", latent_dim=128, num_curves=40, K=8, hidden_dim=256):
    """Auto-detect architecture from checkpoint keys and load."""
    state = torch.load(path, weights_only=True, map_location=device)

    if "curve_decoder.0.weight" in state:
        print(f"  Detected: PerCurveGenerator (v2)")
        model = PerCurveGenerator(latent_dim, num_curves, K, hidden_dim).to(device)
    elif "coarse_decoder.0.weight" in state:
        print(f"  Detected: ConvGenerator (v3)")
        model = ConvGenerator(latent_dim, num_curves, K, hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(state.keys())[:5]}")

    model.load_state_dict(state)
    model.eval()
    return model


class LatentCodes(nn.Module):
    """Per-scene learnable latent codes (auto-decoder)."""

    def __init__(self, num_scenes, latent_dim=128):
        super().__init__()
        self.codes = nn.Parameter(torch.randn(num_scenes, latent_dim) * 0.01)

    def forward(self, indices):
        return self.codes[indices]

    def regularization(self):
        return (self.codes ** 2).mean()


# Alias: SplineGenerator points to the best-performing architecture
# PerCurveGenerator (0.19 drift) outperformed ConvGenerator (0.25 drift)
SplineGenerator = PerCurveGenerator
