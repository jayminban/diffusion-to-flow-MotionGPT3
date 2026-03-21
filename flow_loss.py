import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    def __init__(self, channels, use_mlp_layer=True):
        super().__init__()
        self.channels = channels
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)

        if use_mlp_layer:
            self.mlp = nn.Sequential(
                nn.Linear(channels, channels, bias=True),
                nn.SiLU(),
                nn.Linear(channels, channels, bias=True),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Conv1d(channels, channels, 3, 1, 1),
                nn.ReLU(),
                nn.Linear(channels, channels, bias=True),
            )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class hiddenEmbedWithAtt(nn.Module):
    def __init__(self, input_dim, output_dim=256, output_latent_size=1, proj_first=False,
                 pe_type=None, num_heads=4, mlp_ratio=4., dropout=0.1,
                 activation="gelu", normalize_before=False, **kwargs):
        super().__init__()
        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, '/mnt/data8tb/Documents/models/repo/MotionGPT3')
        from motGPT.archs.operator.cross_attention import TransformerEncoderLayer

        self.proj_first = proj_first
        hidden_dim = output_dim if proj_first else input_dim
        self.cond_proj = nn.Linear(input_dim, output_dim)
        self.hidden_embedding = nn.Parameter(torch.randn(output_latent_size, hidden_dim))

        if pe_type == 'actor':
            from motGPT.archs.operator import PositionalEncoding
            self.hid_pos = PositionalEncoding(hidden_dim, dropout)
        elif pe_type == 'mld':
            from motGPT.archs.operator.position_encoding import build_position_encoding
            self.hid_pos = build_position_encoding(hidden_dim, position_embedding='learned')
        else:
            raise ValueError("Not Support PE type")

        ff_size = int(hidden_dim * mlp_ratio)
        self.hid_encoder_layer = TransformerEncoderLayer(
            hidden_dim, num_heads, ff_size, dropout, activation, normalize_before,
        )

    def forward(self, hidden):
        if self.proj_first:
            hidden = self.cond_proj(hidden)
        bs, seq_len, hidden_dim = hidden.shape
        hidden = hidden.permute(1, 0, 2)
        hidden_dist = torch.tile(self.hidden_embedding[:, None, :], (1, bs, 1))
        hidseq = torch.cat((hidden_dist, hidden), 0)
        hidseq = self.hid_pos(hidseq)
        hidden_dist = self.hid_encoder_layer(hidseq)[:hidden_dist.shape[0]]
        hidden_dist = hidden_dist.permute(1, 0, 2)
        if self.proj_first:
            return hidden_dist
        return self.cond_proj(hidden_dist)


class FlowMatchingNet(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        in_size=None,
        multi_hidden=False,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_size = in_size
        self.multi_hidden = multi_hidden
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        if not self.multi_hidden:
            self.cond_embed = nn.Linear(z_channels, model_channels)
        else:
            self.cond_embed = hiddenEmbedWithAtt(
                input_dim=z_channels,
                output_dim=model_channels,
                output_latent_size=self.in_size,
                proj_first=False,
                pe_type='mld',
                num_heads=4,
                mlp_ratio=4.,
                dropout=0.1  # Same as original diffusion
            )

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels, use_mlp_layer=True))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.final_layer = FinalLayer(model_channels, out_channels)

    def forward(self, x, t, c):
        x = self.input_proj(x)
        t = self.time_embed(t)
        if self.in_size is not None:
            t = t.unsqueeze(1)
        c = self.cond_embed(c)
        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        cond_v, uncond_v = torch.split(model_out, len(model_out) // 2, dim=0)
        half_v = uncond_v + cfg_scale * (cond_v - uncond_v)
        return torch.cat([half_v, half_v], dim=0)


class FlowLoss(nn.Module):
    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_sampling_steps=100,
        target_size=None,
        multi_hidden=False,
        grad_checkpointing=False,
        sigma_min=0.0,  # Minimum noise for optimal transport
        time_sampling='logit_normal',  # 'uniform', 'logit_normal', or 'mode'
        logit_mean=0.0,  # Mean for logit-normal (0.0 centers at t=0.5)
        logit_std=1.0,   # Std for logit-normal (1.0 is SD3 default)
        **kwargs  # Ignore diffusion-specific args like noise_schedule, learn_sigma, etc.
    ):
        super().__init__()

        self.in_size = target_size
        self.in_channels = target_channels
        self.multi_hidden = multi_hidden
        self.num_sampling_steps = int(num_sampling_steps) if isinstance(num_sampling_steps, str) else num_sampling_steps
        self.sigma_min = sigma_min
        self.time_sampling = time_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std

        # Flow network (same architecture as diffusion)
        self.net = FlowMatchingNet(
            in_size=target_size,
            multi_hidden=multi_hidden,
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,  # Predict velocity, same dim as input
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )

    def sample_time(self, batch_size, device):
        if self.time_sampling == 'uniform':
            t = torch.rand(batch_size, device=device)
        elif self.time_sampling == 'logit_normal':
            # Logit-normal: sample from normal, then apply sigmoid
            # This concentrates timesteps around t=0.5 (after sigmoid of 0)
            u = torch.randn(batch_size, device=device) * self.logit_std + self.logit_mean
            t = torch.sigmoid(u)
        elif self.time_sampling == 'mode':
            # Beta distribution concentrated around 0.5
            # Using Beta(2, 2) gives a nice bell curve around 0.5
            t = torch.distributions.Beta(2.0, 2.0).sample((batch_size,)).to(device)
        else:
            raise ValueError(f"Unknown time_sampling: {self.time_sampling}")

        # Clamp to avoid numerical issues at boundaries
        t = t.clamp(1e-5, 1 - 1e-5)
        return t

    def forward(self, target, z, mask=None):
        batch_size = target.shape[0]
        device = target.device

        # Sample time according to chosen strategy
        t = self.sample_time(batch_size, device)

        # Sample x_0 ~ N(0, I) (noise/source distribution)
        x_0 = torch.randn_like(target)

        # Optimal transport / rectified flow path: x_t = (1-t)*x_0 + t*x_1
        # This is the straight-line path from noise to data
        if target.dim() == 3:
            t_expand = t[:, None, None]
        else:
            t_expand = t[:, None]

        # Linear interpolation (rectified flow)
        x_t = (1 - t_expand) * x_0 + t_expand * target

        # Optional: add small noise for stability
        if self.sigma_min > 0:
            x_t = x_t + self.sigma_min * torch.randn_like(x_t)

        # True velocity: v = x_1 - x_0 (constant along the straight path)
        v_true = target - x_0

        # Predict velocity
        v_pred = self.net(x_t, t, z)

        # MSE loss (this is the rectified flow objective)
        loss = (v_pred - v_true) ** 2

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()

        return loss.mean()

    @torch.no_grad()
    def sample(self, z, temperature=1.0, cfg=1.0, num_steps=None):
        if num_steps is None:
            num_steps = self.num_sampling_steps

        # print(f"[DEBUG] FlowLoss.sample(Euler) num_steps={num_steps}, cfg={cfg}")
        device = z.device

        # Handle CFG: z is doubled [cond; uncond]
        if cfg != 1.0:
            bsz = z.shape[0] // 2
        else:
            bsz = z.shape[0]

        # Initialize x_0 ~ N(0, I)
        if self.in_size is not None:
            x = torch.randn(bsz, self.in_size, self.in_channels, device=device) * temperature
        else:
            x = torch.randn(bsz, self.in_channels, device=device) * temperature

        # For CFG, duplicate x
        if cfg != 1.0:
            x = torch.cat([x, x], dim=0)

        # Time steps from 0 to 1
        dt = 1.0 / num_steps

        # Euler integration
        for i in range(num_steps):
            t = torch.ones(x.shape[0], device=device) * (i * dt)
            # print(f"[DEBUG] time step: {t[0].item():.4f}")
            # print(f"[DEBUG-during-step] Sampling step euler being called")
            if cfg != 1.0:
                v = self.net.forward_with_cfg(x, t, z, cfg)
            else:
                v = self.net(x, t, z)

            x = x + v * dt
        # print(f"[DEBUG--One-example-done] Sampling step euler being called")

        # Return full tensor (both cond and uncond) to match DiffLoss behavior
        # The caller (sample_tokens) will chunk it to get only conditioned samples
        return x

    @torch.no_grad()
    def sample_rk4(self, z, temperature=1.0, cfg=1.0, num_steps=None):
        if num_steps is None:
            num_steps = self.num_sampling_steps

        # print(f"[DEBUG] FlowLoss.sample_rk4() num_steps={num_steps}, cfg={cfg}")
        device = z.device

        if cfg != 1.0:
            bsz = z.shape[0] // 2
        else:
            bsz = z.shape[0]

        if self.in_size is not None:
            x = torch.randn(bsz, self.in_size, self.in_channels, device=device) * temperature
        else:
            x = torch.randn(bsz, self.in_channels, device=device) * temperature

        if cfg != 1.0:
            x = torch.cat([x, x], dim=0)

        dt = 1.0 / num_steps

        def get_velocity(x, t_val):
            t = torch.ones(x.shape[0], device=device) * t_val
            if cfg != 1.0:
                return self.net.forward_with_cfg(x, t, z, cfg)
            else:
                return self.net(x, t, z)

        # RK4 integration
        for i in range(num_steps):
            # print(f"[DEBUG--during-step] Sampling step rk4 being called")

            t = i * dt

            k1 = get_velocity(x, t)
            k2 = get_velocity(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = get_velocity(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = get_velocity(x + dt * k3, t + dt)

            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
        # print(f"[DEBUG--One-example-done] Sampling step rk4 being called")
        # Return full tensor (both cond and uncond) to match DiffLoss behavior
        # The caller (sample_tokens) will chunk it to get only conditioned samples
        return x
