"""Amplitude-preserving periodic Fourier neural operator (real policy vector)."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NetworkConfig:
    nx: int = 96
    width: int = 29
    action_grid: int = 96
    action_channels: int = 1
    modes: int = 6
    layers: int = 4
    radial_bins: int = 14
    bilinear_channels: int = 16
    head_bands: int = 6
    nyquist_gate: bool = False
    nyquist_cutoff: float = 42.0
    nyquist_tau: float = 5e-4


def _radial_knots(count, rho_max, device, dtype, alpha=2.0):
    u = torch.linspace(0.0, 1.0, count, device=device, dtype=dtype)
    if alpha <= 0:
        return u * rho_max
    return rho_max * torch.expm1(alpha * u) / math.expm1(alpha)


def _tent_partition(rho, knots):
    """Continuous piecewise-linear partition of unity over [0, rho_max]."""
    count = len(knots)
    out = rho.new_zeros((count, *rho.shape))
    for m in range(count):
        if m == 0:
            right = (knots[1] - rho) / (knots[1] - knots[0]).clamp(min=1e-12)
            out[m] = right.clamp(0.0, 1.0)
        elif m == count - 1:
            left = (rho - knots[m - 1]) / (knots[m] - knots[m - 1]).clamp(min=1e-12)
            out[m] = left.clamp(0.0, 1.0)
        else:
            left = (rho - knots[m - 1]) / (knots[m] - knots[m - 1]).clamp(min=1e-12)
            right = (knots[m + 1] - rho) / (knots[m + 1] - knots[m]).clamp(min=1e-12)
            out[m] = torch.minimum(left, right).clamp(0.0, 1.0)
    total = out.sum(0, keepdim=True).clamp(min=1e-12)
    return out / total


def enforce_rfft2_hermitian(z, n, h):
    """Enforce exact real-field conjugate symmetry on self-conjugate rfft2 columns."""
    out = z.clone()
    cols = [0]
    if h % 2 == 0:
        cols.append(h // 2)
    for col in cols:
        out[..., 0, col] = out[..., 0, col].real.to(out.dtype)
        if n % 2 == 0:
            out[..., n // 2, col] = out[..., n // 2, col].real.to(out.dtype)
        half = (n + 1) // 2
        if half > 1:
            pos = out[..., 1:half, col]
            neg = torch.flip(out[..., n - half + 1 :, col], dims=[-1])
            sym_pos = 0.5 * (pos + torch.conj(neg))
            sym_neg = torch.flip(torch.conj(sym_pos), dims=[-1])
            out[..., 1:half, col] = sym_pos
            out[..., n - half + 1 :, col] = sym_neg
    return out


class SpectralConv2d(nn.Module):
    """Low-mode 2D table plus full-2D-spectrum radial-harmonic filter bank."""

    def __init__(self, width, modes, radial_bins=14):
        super().__init__()
        self.modes = modes
        self.low_modes = max(1, modes - 1) if radial_bins > 0 else modes
        self.radial_bins = radial_bins
        shape = (width, width, self.low_modes, self.low_modes, 2)
        self.positive = nn.Parameter(torch.randn(shape) / width)
        self.negative = nn.Parameter(torch.randn(shape) / width)
        if radial_bins > 0:
            self.radial = nn.Parameter(
                torch.randn(width, width, radial_bins, 3) / (width * math.sqrt(3.0))
            )
        else:
            self.register_parameter("radial", None)
        self._basis_cache = {}

    def _radial_basis(self, n, h, device, dtype):
        key = (n, h, device, dtype)
        cached = self._basis_cache.get(key)
        if cached is not None:
            return cached
        kx = torch.fft.fftfreq(n, device=device, dtype=dtype) * n
        ky = torch.fft.rfftfreq(h, device=device, dtype=dtype) * h
        rho = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
        rho_max = max(float(rho.max()), 1.0)
        knots = _radial_knots(self.radial_bins, rho_max, device, dtype)
        tent = _tent_partition(rho, knots)
        ikx = kx.clone()
        iky = ky.clone()
        if n % 2 == 0:
            ikx[n // 2] = 0.0
        if h % 2 == 0:
            iky[-1] = 0.0
        denom = rho.clamp(min=1.0)
        px = ikx[:, None] / denom
        py = iky[None, :] / denom
        basis = (tent, tent * px[None], tent * py[None])
        self._basis_cache[key] = basis
        return basis

    def forward_spectrum(self, z, n, h):
        mx = min(self.low_modes, n // 2)
        my = min(self.low_modes, h // 2)
        if self.radial is not None:
            tent, tent_px, tent_py = self._radial_basis(n, h, z.device, self.radial.dtype)
            real_mult = torch.einsum("ior,rxy->ioxy", self.radial[..., 0], tent)
            imag_mult = torch.einsum("ior,rxy->ioxy", self.radial[..., 1], tent_px) + torch.einsum(
                "ior,rxy->ioxy", self.radial[..., 2], tent_py
            )
            w_full = torch.complex(real_mult, imag_mult).to(z.dtype)
            out = torch.einsum("bixy,ioxy->boxy", z, w_full)
        else:
            out = z.new_zeros((len(z), z.shape[1], n, h // 2 + 1))
        for region, weights in (
            (slice(0, mx), self.positive),
            (slice(-mx, None), self.negative),
        ):
            parts = weights[:, :, :mx, :my]
            # Flat policy assignment can give an odd real storage offset.
            # Construct complex values explicitly instead of view_as_complex.
            w = torch.complex(parts[..., 0], parts[..., 1]).to(z.dtype)
            out[:, :, region, :my] = out[:, :, region, :my] + torch.einsum(
                "bixy,ioxy->boxy", z[:, :, region, :my], w
            )
        if self.radial is None:
            return out
        return enforce_rfft2_hermitian(out, n, h)

    def forward(self, x):
        z = torch.fft.rfft2(x, norm="ortho")
        n, h = x.shape[-2:]
        out = self.forward_spectrum(z, n, h)
        return torch.fft.irfft2(out, s=(n, h), norm="ortho")


def fourier_upsample_2x(u):
    """Exact 2x trigonometric interpolation with symmetric Nyquist splitting."""
    n, h = u.shape[-2:]
    u_hat = torch.fft.rfft2(u, norm="backward")
    pad_hat = u_hat.new_zeros((*u.shape[:-2], 2 * n, h + 1))
    nx_pos = (n + 1) // 2
    nx_neg = (n - 1) // 2
    pad_hat[..., :nx_pos, : h // 2 + 1] = u_hat[..., :nx_pos, :]
    if nx_neg > 0:
        pad_hat[..., -nx_neg:, : h // 2 + 1] = u_hat[..., -nx_neg:, :]
    if n % 2 == 0:
        nyq_x = 0.5 * u_hat[..., n // 2, :]
        pad_hat[..., n // 2, : h // 2 + 1] = nyq_x
        pad_hat[..., 2 * n - n // 2, : h // 2 + 1] = nyq_x
    if h % 2 == 0:
        pad_hat[..., :, h // 2] *= 0.5
    return torch.fft.irfft2(pad_hat * 4.0, s=(2 * n, 2 * h), norm="backward")


def fourier_restrict_2x(u_fine, n, h):
    """Exact 2x spectral restriction back to (n, h) with Hermitian Nyquist handling."""
    prod_hat = torch.fft.rfft2(u_fine, norm="backward") * 0.25
    out_hat = prod_hat.new_zeros((*u_fine.shape[:-2], n, h // 2 + 1))
    nx_pos = (n + 1) // 2
    nx_neg = (n - 1) // 2
    out_hat[..., :nx_pos, :] = prod_hat[..., :nx_pos, : h // 2 + 1]
    if nx_neg > 0:
        out_hat[..., -nx_neg:, :] = prod_hat[..., -nx_neg:, : h // 2 + 1]
    if n % 2 == 0:
        out_hat[..., n // 2, :] = 0.0
    if h % 2 == 0:
        out_hat[..., :, h // 2] = 0.0
    out_hat = enforce_rfft2_hermitian(out_hat, n, h)
    return torch.fft.irfft2(out_hat, s=(n, h), norm="backward")


def dealiased_quadratic_product(a, b):
    """Explicit dealiased quadratic product Q(a, b) = R_{2N->N}[(I_{N->2N} a)(I_{N->2N} b)]."""
    n, h = a.shape[-2:]
    a_fine = fourier_upsample_2x(a)
    b_fine = fourier_upsample_2x(b)
    return fourier_restrict_2x(a_fine * b_fine, n, h)


class DealiasedBilinear2d(nn.Module):
    """Paired explicit dealiased product Q(a, b) and local product a * b."""

    def __init__(self, width, bilinear_channels=16):
        super().__init__()
        c = min(bilinear_channels, width)
        self.bilinear_channels = c
        self.left = nn.Conv2d(width, c, 1, bias=True)
        self.right = nn.Conv2d(width, c, 1, bias=True)
        self.out = nn.Conv2d(2 * c, width, 1, bias=True)

    def forward(self, x):
        a = self.left(x)
        b = self.right(x)
        a = a * torch.rsqrt(a.square().mean(dim=(-2, -1), keepdim=True) + 1.0)
        b = b * torch.rsqrt(b.square().mean(dim=(-2, -1), keepdim=True) + 1.0)
        q_ab = dealiased_quadratic_product(a, b)
        ab = a * b
        return self.out(torch.cat((q_ab, ab), dim=1))


class BandSplit2d(nn.Module):
    """Post-nonlinearity, pre-head radial wavenumber band partition of feature channels."""

    def __init__(self, width, num_bands=6, nyquist_cutoff=42.0, nyquist_tau=5e-4):
        super().__init__()
        self.width = width
        self.num_bands = max(1, min(num_bands, width))
        self.nyquist_cutoff = float(nyquist_cutoff)
        self.nyquist_tau = float(nyquist_tau)
        self.channel_bands = [
            min((c * self.num_bands) // width, self.num_bands - 1) for c in range(width)
        ]
        self.closure_bands = max(1, self.num_bands - 2) if self.num_bands >= 6 else self.num_bands
        self.closure_channels = sum(1 for b in self.channel_bands if b < self.closure_bands)
        self._mask_cache = {}
        self._tail_cache = {}

    def band_channels(self):
        groups = [[] for _ in range(self.num_bands)]
        for c, b in enumerate(self.channel_bands):
            groups[b].append(c)
        return [g for g in groups if g]

    def _tail_parseval_weights(self, n, h, device, dtype):
        key = (n, h, device, dtype)
        cached = self._tail_cache.get(key)
        if cached is not None:
            return cached
        kx = torch.fft.fftfreq(n, device=device, dtype=dtype) * n
        ky = torch.fft.rfftfreq(h, device=device, dtype=dtype) * h
        rho = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
        cutoff = self.nyquist_cutoff * (float(n) / 96.0)
        mask = (rho >= cutoff).to(dtype)
        mult = torch.full((n, h // 2 + 1), 2.0, device=device, dtype=dtype)
        mult[:, 0] = 1.0
        if h % 2 == 0:
            mult[:, -1] = 1.0
        weights = mask * mult / float(n * h)
        self._tail_cache[key] = weights
        return weights

    def nyquist_gate(self, x):
        """Exact Parseval spatial mean-square high-wavenumber energy homogeneity gate."""
        if self.num_bands < 6:
            return x.new_ones((len(x), 1, 1, 1))
        n, h = x.shape[-2:]
        weights = self._tail_parseval_weights(n, h, x.device, x.dtype)
        z_sq = torch.fft.rfft2(x, norm="ortho").abs().square()
        e_tail = (z_sq * weights).sum(dim=(-2, -1), keepdim=True).mean(dim=-3, keepdim=True)
        tau = self.nyquist_tau
        return (e_tail * e_tail) / (e_tail * e_tail + tau * tau)

    def _channel_transfer(self, n, h, device, dtype):
        key = (n, h, device, dtype)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        kx = torch.fft.fftfreq(n, device=device, dtype=dtype) * n
        ky = torch.fft.rfftfreq(h, device=device, dtype=dtype) * h
        rho = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
        rho_max = max(float(rho.max()), 1.0)
        if self.num_bands == 1:
            transfer = torch.ones((self.width, n, h // 2 + 1), device=device, dtype=dtype)
        else:
            knots = _radial_knots(self.num_bands, rho_max, device, dtype)
            tent = _tent_partition(rho, knots)
            idx = torch.tensor(self.channel_bands, device=device, dtype=torch.long)
            transfer = tent[idx]
        self._mask_cache[key] = transfer
        return transfer

    def forward(self, x):
        if self.num_bands <= 1:
            return x
        n, h = x.shape[-2:]
        z = torch.fft.rfft2(x, norm="ortho")
        transfer = self._channel_transfer(n, h, x.device, x.dtype)
        return torch.fft.irfft2(z * transfer[None], s=(n, h), norm="ortho")


class PeriodicFNO(nn.Module):
    """Periodic Fourier neural operator with optional radial-harmonic and dealiased bilinear blocks.

    With width 29, 6 modes, and 4 blocks, the closure has 489,057 trainable parameters for
    Cahn-Hilliard and Gray-Scott, 490,027 for Kolmogorov Navier-Stokes, and 490,057 for
    2D Guide-Field MHD. The zero-initialized head aligns the initial residual with zero.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        w = config.width
        if config.action_grid != config.nx:
            raise ValueError("full-grid component required")
        self.legacy = (
            config.radial_bins == 0 and config.bilinear_channels == 0 and config.head_bands <= 1
        )
        self.lift = nn.Conv2d(8, w, 1, bias=True)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(w, config.modes, config.radial_bins) for _ in range(config.layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv2d(w, w, 1, bias=True) for _ in range(config.layers)]
        )
        if self.legacy:
            self.bilinear = None
        else:
            self.bilinear = nn.ModuleList(
                [DealiasedBilinear2d(w, config.bilinear_channels) for _ in range(config.layers)]
            )
        self.project = nn.Conv2d(w, w, 1, bias=True)
        self.band_split = BandSplit2d(
            w,
            1 if self.legacy else config.head_bands,
            nyquist_cutoff=config.nyquist_cutoff,
            nyquist_tau=config.nyquist_tau,
        )
        self.head = nn.Conv2d(w, config.action_channels, 1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _raw_trunk_features(self, x):
        x = self.lift(x)
        if self.legacy:
            for spectral, pointwise in zip(self.spectral, self.pointwise, strict=True):
                x = F.gelu(spectral(x) + pointwise(x))
            return F.gelu(self.project(x))
        for spectral, pointwise, bilinear in zip(
            self.spectral[:-1],
            self.pointwise[:-1],
            self.bilinear[:-1],
            strict=True,
        ):
            x = x + F.gelu(spectral(x) + pointwise(x) + bilinear(x))
        nonlinear = self.pointwise[-1](F.gelu(self.project(x)))
        linear_spec = self.spectral[-1](x)
        dealiased_quad = self.bilinear[-1](x)
        return self.band_split(nonlinear + linear_spec + dealiased_quad)

    def trunk_features(self, x):
        feat = self._raw_trunk_features(x)
        if not self.legacy and self.config.nyquist_gate and self.band_split.num_bands >= 6:
            gamma = self.band_split.nyquist_gate(x)
            c_cut = self.band_split.closure_channels
            feat = torch.cat((feat[:, :c_cut], gamma * feat[:, c_cut:]), dim=1)
        return feat

    def split_head(self, x, detach_tail_trunk=False):
        feat = self._raw_trunk_features(x)
        if not self.legacy and self.config.nyquist_gate and self.band_split.num_bands >= 6:
            gamma = self.band_split.nyquist_gate(x)
            c_cut = self.band_split.closure_channels
            tail_feat = feat[:, c_cut:].detach() if detach_tail_trunk else feat[:, c_cut:]
            closure = F.conv2d(feat[:, :c_cut], self.head.weight[:, :c_cut], self.head.bias)
            tail = F.conv2d(gamma * tail_feat, self.head.weight[:, c_cut:], None)
            return closure, tail
        out = self.head(feat)
        return out, torch.zeros_like(out)

    def forward(self, x):
        return self.head(self.trunk_features(x))
