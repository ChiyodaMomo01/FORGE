from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDilatedSpectralBlock(nn.Module):
    """2D residual block combining local 3x3 conv, dilated 3x3 conv, and low-mode spectral gating."""

    def __init__(self, channels: int, dilation: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(16, channels)
        self.conv_local = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=4)
        self.conv_dilated = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=4
        )
        self.norm2 = nn.GroupNorm(16, channels)
        self.pw1 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.pw2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.20))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(x))
        h_cat = torch.cat([self.conv_local(h), self.conv_dilated(h)], dim=1)
        h = F.gelu(self.norm2(self.pw1(h_cat)))
        h = self.pw2(h) * self.se(h)
        return x + self.res_scale * h


class ForgePOMNet(nn.Module):
    """3D-aware closure network G_theta(x_k) with a 256-dim trunk and 257-dim linear readout head."""

    def __init__(
        self,
        in_channels: int = 161,
        hidden_channels: int = 256,
        out_channels: int = 60,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(16, hidden_channels),
            nn.GELU(),
        )
        dilations = [1, 2, 3, 4, 1, 2, 3, 4][:num_blocks]
        self.blocks = nn.ModuleList(
            [ResidualDilatedSpectralBlock(hidden_channels, dilation=d) for d in dilations]
        )
        self.trunk_norm = nn.GroupNorm(16, hidden_channels)
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True)
        self.reset_head_zero()

    def reset_head_zero(self) -> None:
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def extract_trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return trunk feature map phi_k in R^{B x 256 x 90 x 82}."""
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        phi = F.gelu(self.trunk_norm(h))
        return phi[0] if single else phi

    def forward_from_features(self, phi: torch.Tensor) -> torch.Tensor:
        single = phi.dim() == 3
        if single:
            phi = phi.unsqueeze(0)
        out = self.head(phi)
        return out[0] if single else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.extract_trunk_features(x)
        return self.forward_from_features(phi)

    @torch.no_grad()
    def get_head_matrix(self) -> torch.Tensor:
        """Return augmented readout matrix W_aug = [W, b] in R^{60 x 257}."""
        w = self.head.weight.data.squeeze(-1).squeeze(-1)  # (60, 256)
        b = self.head.bias.data.unsqueeze(1)  # (60, 1)
        return torch.cat([w, b], dim=1).clone()

    @torch.no_grad()
    def set_head_matrix(self, w_aug: torch.Tensor) -> None:
        """Set augmented readout matrix W_aug = [W, b] in R^{60 x 257}."""
        self.head.weight.data.copy_(w_aug[:, : self.hidden_channels].unsqueeze(-1).unsqueeze(-1))
        if self.head.bias is not None:
            self.head.bias.data.copy_(w_aug[:, self.hidden_channels])

    @torch.no_grad()
    def get_flat_params(self) -> torch.Tensor:
        """Return all network parameters (trunk + head) as a single 1D vector."""
        return torch.cat([p.data.reshape(-1) for p in self.parameters()], dim=0).clone()

    @torch.no_grad()
    def set_flat_params(self, flat: torch.Tensor) -> None:
        """Set all network parameters (trunk + head) from a single 1D vector."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset : offset + numel].reshape_as(p.data))
            offset += numel

    @torch.no_grad()
    def head_matrix_to_flat_delta(self, delta_w_aug: torch.Tensor) -> torch.Tensor:
        """Embed a (60, 257) head matrix perturbation into the full flat parameter space."""
        flat = torch.zeros_like(self.get_flat_params())
        offset = 0
        for name, p in self.named_parameters():
            numel = p.numel()
            if name == "head.weight":
                flat[offset : offset + numel].copy_(
                    delta_w_aug[:, : self.hidden_channels].reshape(-1)
                )
            elif name == "head.bias":
                flat[offset : offset + numel].copy_(
                    delta_w_aug[:, self.hidden_channels].reshape(-1)
                )
            offset += numel
        return flat

    @torch.no_grad()
    def scale_head_parameters(self, beta: float) -> None:
        """Multiply only the readout head parameters by scalar beta."""
        self.head.weight.data.mul_(float(beta))
        if self.head.bias is not None:
            self.head.bias.data.mul_(float(beta))

    @torch.no_grad()
    def head_body_index_masks(
        self, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return boolean masks (head_mask, body_mask) over the flattened parameter vector."""
        dev = device if device is not None else next(self.parameters()).device
        total = sum(p.numel() for p in self.parameters())
        head_mask = torch.zeros(total, dtype=torch.bool, device=dev)
        offset = 0
        for name, p in self.named_parameters():
            numel = p.numel()
            if name.startswith("head."):
                head_mask[offset : offset + numel] = True
            offset += numel
        return head_mask, ~head_mask

    @torch.no_grad()
    def build_mixed_head_body_basis(
        self,
        num_directions: int = 16,
        seed: int = 0,
    ) -> list[torch.Tensor]:
        """Build `num_directions` orthonormal mixed head/body directions matching 2D `common.basis`.

        Head indices receive Q_h / sqrt(2) and trunk indices receive Q_b / sqrt(2), where Q_h and
        Q_b are reduced QR factors with deterministic positive R diagonal signs generated from `seed`.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        head_mask_cpu, body_mask_cpu = self.head_body_index_masks(device=torch.device("cpu"))
        n_total = int(head_mask_cpu.numel())
        n_head = int(head_mask_cpu.sum().item())
        n_body = int(body_mask_cpu.sum().item())
        if num_directions > min(n_head, n_body):
            raise ValueError(
                f"num_directions={num_directions} exceeds min(n_head={n_head}, n_body={n_body})"
            )

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        u_cpu = torch.zeros((n_total, num_directions), dtype=torch.float32)
        inv_sqrt2 = float(0.5**0.5)
        for mask, dim_len in ((head_mask_cpu, n_head), (body_mask_cpu, n_body)):
            raw = torch.randn((dim_len, num_directions), generator=gen, dtype=torch.float32)
            q, r_mat = torch.linalg.qr(raw, mode="reduced")
            diag_sign = torch.where(
                torch.diagonal(r_mat) >= 0.0,
                torch.ones(num_directions, dtype=torch.float32),
                -torch.ones(num_directions, dtype=torch.float32),
            )
            q = q * diag_sign.unsqueeze(0)
            u_cpu[mask] = q * inv_sqrt2

        u_dev = u_cpu.to(device=device, dtype=dtype)
        return [u_dev[:, m].contiguous() for m in range(num_directions)]
