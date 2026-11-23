"""Complex spectral and local response operators with explicit adjoints."""

import numpy as np
import torch

from .fitting import fit_response


class SplitTangent:
    """Complex Fourier multipliers plus public-quadratic/own-G modulation.

    Imaginary x/y basis functions obey real-field Hermitian symmetry. The full
    map need not be self-adjoint; its norm is bounded by max|spectrum|+max|local|.
    Contexts are computed once from frozen public anchors, never differentiated
    through F/H or recomputed inside the fitted-linear-model rollout.
    """

    def __init__(
        self,
        n,
        model,
        device="cpu",
        *,
        no_modulation=False,
        one_step_only=False,
        step_gain_limit=1.05,
        auto_step_gain=False,
    ):
        if n < 2 or not np.isfinite(step_gain_limit) or step_gain_limit <= 0:
            raise ValueError("positive finite step gain and grid size >= 2 required")
        self.no_modulation = bool(no_modulation)
        self.one_step_only = bool(one_step_only)
        self.step_gain_limit = float(step_gain_limit)
        self.auto_step_gain = bool(auto_step_gain)
        self.n = n
        self.model = model
        self.device = device
        self.weights = None
        self.fit_keys = set()
        self._operators = None
        self.context_cache = {}
        self.context_scale = None
        self.context_mean = None
        ky = torch.fft.fftfreq(n, device=device) * n
        kx = torch.fft.rfftfreq(n, device=device) * n
        squared = (ky[:, None] ** 2 + kx[None, :] ** 2).round().long().clamp(max=150)
        labels = torch.unique(squared)
        self.shell = torch.searchsorted(labels, squared.contiguous())
        self.shell_values = labels.cpu().tolist()
        self.shell_count = len(labels)
        radius = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).clamp(min=1)
        # Odd multipliers vanish at self-conjugate/Nyquist coordinates.
        ix = kx.clone()
        iy = ky.clone()
        sq_mask = torch.ones((n, n // 2 + 1), device=device, dtype=torch.float32)
        if n % 2 == 0:
            ix[-1] = 0
            iy[n // 2] = 0
            sq_mask[:, -1] = 0.0
            sq_mask[n // 2, :] = 0.0
        r2 = (iy[:, None] ** 2 + ix[None, :] ** 2) * sq_mask
        k_half = max(float(n // 2), 1.0)
        s_norm = (r2 / (k_half**2)) * sq_mask
        self.poly_full = torch.stack(
            (
                sq_mask.to(torch.complex64),
                s_norm.to(torch.complex64),
                (s_norm**2).to(torch.complex64),
                ((1j * ix[None, :] / k_half) * sq_mask).expand(n, n // 2 + 1).to(torch.complex64),
                ((1j * iy[:, None] / k_half) * sq_mask).expand(n, n // 2 + 1).to(torch.complex64),
            )
        )
        hk_mask = ((r2 > 50.0).float() * sq_mask).to(torch.complex64)
        self.poly_hk = self.poly_full * hk_mask[None]
        self.poly_count = len(self.poly_hk)
        px = 1j * ix[None, :].expand_as(radius) / radius
        py = 1j * iy[:, None].expand_as(radius) / radius
        gx = 1j * ix[None, :].expand_as(radius) / float(n)
        gy = 1j * iy[:, None].expand_as(radius) / float(n)
        self.local_ops = torch.stack((px, py, gx, gy)).to(torch.complex64)
        self.local_op_norms = self.local_ops.abs().amax(dim=(1, 2))
        self.local_blocks = 1 + 2 * len(self.local_ops)
        basis = []
        for j in range(self.shell_count):
            mask = (self.shell == j).float()
            basis.extend(
                (
                    mask.to(torch.complex64),
                    (mask * px).to(torch.complex64),
                    (mask * py).to(torch.complex64),
                )
            )
        basis.extend(list(self.poly_hk))
        self.basis = torch.stack(basis)
        self.spectral_count = len(basis)
        self.spectral_features = 2 * self.spectral_count
        self.state_channels = 1
        self.action_channels = 1
        self.input_channels = 2
        self.context_channels = None
        self.features_count = None

    def raw_context(self, anchors, batch=4):
        """All public linear/quadratic channel terms and own G output."""
        shape = anchors.shape
        x = torch.as_tensor(anchors, device=self.device).flatten(0, 1)
        parts = []
        self.model.eval()
        with torch.no_grad():
            for current in x.split(batch):
                quadratic = [
                    current[:, i : i + 1] * current[:, j : j + 1]
                    for i in range(current.shape[1])
                    for j in range(i, current.shape[1])
                ]
                parts.append(torch.cat((current, *quadratic, self.model(current)), dim=1).detach())
        return torch.cat(parts).reshape(*shape[:2], -1, *shape[-2:])

    def prepare_contexts(self, traces):
        unique = list(dict.fromkeys(traces))
        raw = [self.raw_context(t.anchors) for t in unique]
        if self.context_scale is None:
            first = unique[0]
            self.state_channels = int(getattr(first, "field_channels", 1))
            self.action_channels = int(getattr(first, "action_channels", 1))
            self.input_channels = self.state_channels + self.action_channels
            self.spectral_features = self.input_channels * self.spectral_count
            all_context = torch.cat(raw, dim=1)
            dims = (0, 1, 3, 4)
            peak = all_context.abs().amax(dims)
            self.context_mean = torch.zeros_like(peak)
            scale_raw = peak * np.sqrt(self.shell_count)
            self.context_scale = torch.where(
                peak > 0, scale_raw.clamp(min=1e-30), torch.full_like(scale_raw, 1e-6)
            )
            self.context_channels = len(peak)
            self.features_count = (
                self.spectral_features
                + self.local_blocks * self.input_channels * self.context_channels
            )
            self.state_ids = [
                idx
                for base in range(0, self.features_count, self.input_channels)
                for idx in range(base, base + self.state_channels)
            ]
            self.force_ids = [
                idx
                for base in range(0, self.features_count, self.input_channels)
                for idx in range(base + self.state_channels, base + self.input_channels)
            ]
        for trace, value in zip(unique, raw, strict=True):
            # Fit-center peak scaling only; all original field means remain.
            self.context_cache[trace] = value / self.context_scale[None, None, :, None, None]

    def contexts(self, trace):
        if trace not in self.context_cache:
            self.prepare_contexts([trace])
        return self.context_cache[trace]

    def features(self, context, dx, forcing):
        v = torch.cat((dx, forcing), dim=1)
        freq = torch.fft.rfft2(v, norm="ortho")
        spectral = torch.fft.irfft2(
            freq[:, None] * self.basis[None, :, None], s=(self.n, self.n), norm="ortho"
        ).flatten(1, 2)
        if self.no_modulation:
            local = spectral.new_zeros(
                v.shape[0],
                self.local_blocks * self.context_channels * self.input_channels,
                self.n,
                self.n,
            )
            return torch.cat((spectral, local), dim=1)
        s4 = self.local_ops.to(freq)
        op_v = torch.fft.irfft2(s4[None, :, None] * freq[:, None], s=(self.n, self.n), norm="ortho")
        loc0 = (context[:, :, None] * v[:, None]).flatten(1, 2)
        loc_in = (context[:, None, :, None] * op_v[:, :, None]).flatten(1, 3)
        cv = context[:, :, None] * v[:, None]
        f_cv = torch.fft.rfft2(cv, norm="ortho")
        loc_out = torch.fft.irfft2(
            s4[None, :, None, None] * f_cv[:, None],
            s=(self.n, self.n),
            norm="ortho",
        ).flatten(1, 3)
        return torch.cat((spectral, loc0, loc_in, loc_out), dim=1)

    def symbols(self, weights):
        w = weights[:, :, None] if weights.ndim == 2 else weights
        spec = w[:, : self.spectral_features].reshape(
            -1, self.spectral_count, self.input_channels, self.state_channels
        )
        return torch.einsum("tkio,khw->toihw", spec.to(self.basis), self.basis)

    def bound_tensor(self, weights, contexts):
        w = weights[:, :, None] if weights.ndim == 2 else weights
        if self.no_modulation:
            w = torch.cat(
                (
                    w[:, : self.spectral_features],
                    torch.zeros_like(w[:, self.spectral_features :]),
                ),
                dim=1,
            )
        spectrum = self.symbols(w)
        local = w[:, self.spectral_features :].reshape(
            -1,
            self.local_blocks,
            self.context_channels,
            self.input_channels,
            self.state_channels,
        )
        modulation = (
            torch.einsum("tbjhw,tjio->tboihw", contexts, local[:, 0])
            .permute(0, 2, 3, 1, 4, 5)
            .flatten(3)
        )
        lower = modulation.min(3).values
        upper = modulation.max(3).values
        shifts = (torch.zeros_like(lower), (lower + upper) / 2, lower, upper)
        bounds = []
        for shift in shifts:
            spec_b = (spectrum + shift[:, :, :, None, None]).abs().flatten(3).max(3).values
            loc_b = torch.maximum((lower - shift).abs(), (upper - shift).abs())
            bounds.append(spec_b + loc_b)
        block = torch.stack(bounds).min(0).values
        if not self.no_modulation:
            mod_in = (
                torch.einsum("tbjhw,tdjio->tdboihw", contexts, local[:, 1:5])
                .abs()
                .amax(dim=(2, 5, 6))
            )
            mod_out = (
                torch.einsum("tbjhw,tdjio->tdboihw", contexts, local[:, 5:9])
                .abs()
                .amax(dim=(2, 5, 6))
            )
            op_norms = self.local_op_norms.to(block)[None, :, None, None]
            block = block + ((mod_in + mod_out) * op_norms).sum(1)

        def op_norm2(mat):
            if mat.shape[1] == 1:
                return mat.square().sum(2).clamp(min=1e-30).sqrt().squeeze(1)
            if mat.shape[1] == 2:
                g00 = mat[:, 0].square().sum(1)
                g11 = mat[:, 1].square().sum(1)
                g01 = (mat[:, 0] * mat[:, 1]).sum(1)
                disc = ((g00 - g11).square() + 4 * g01.square() + 1e-30).sqrt()
                return ((g00 + g11 + disc) / 2).clamp(min=1e-30).sqrt()
            return mat.square().sum((1, 2)).clamp(min=1e-30).sqrt()

        state_bound = op_norm2(block[:, :, : self.state_channels])
        force_bound = op_norm2(block[:, :, self.state_channels :])
        return torch.stack((state_bound, force_bound), dim=1)

    def norm_bounds(self, contexts):
        return self.bound_tensor(self.weights, contexts).max(0).values.detach().cpu().tolist()

    def fit(self, pairs, ridge=1e-6):
        return fit_response(self, pairs, ridge)

    def operators(self, interval):
        if self._operators is None:
            w = self.weights[:, :, None] if self.weights.ndim == 2 else self.weights
            self._operators = (
                self.symbols(w),
                w[:, self.spectral_features :].reshape(
                    -1,
                    self.local_blocks,
                    self.context_channels,
                    self.input_channels,
                    self.state_channels,
                ),
            )
        spectral, local = (v[interval] for v in self._operators)
        if self.no_modulation:
            local = torch.zeros_like(local)
        return spectral, local

    def apply(self, context, dx, forcing, interval=0):
        v = torch.cat((dx, forcing), dim=1)
        spectral, local = self.operators(interval)
        freq = torch.fft.rfft2(v, norm="ortho")
        value = torch.fft.irfft2(
            torch.einsum("oihw,bihw->bohw", spectral, freq),
            s=(self.n, self.n),
            norm="ortho",
        )
        if self.no_modulation:
            return value
        s4 = self.local_ops.to(freq)
        m = torch.einsum("mjio,bjhw->mboihw", local.to(v), context)
        op_v = torch.fft.irfft2(s4[None, :, None] * freq[:, None], s=(self.n, self.n), norm="ortho")
        spatial = torch.einsum("boihw,bihw->bohw", m[0], v) + torch.einsum(
            "dboihw,bdihw->bohw", m[1:5], op_v
        )
        flux_out = torch.einsum("dboihw,bihw->bodhw", m[5:9], v)
        spec_out = (torch.fft.rfft2(flux_out, norm="ortho") * s4[None, None]).sum(2)
        return value + spatial + torch.fft.irfft2(spec_out, s=(self.n, self.n), norm="ortho")

    def adjoint(self, context, q, interval=0):
        spectral, local = self.operators(interval)
        freq = torch.fft.rfft2(q, norm="ortho")
        value = torch.fft.irfft2(
            torch.einsum("oihw,bohw->bihw", spectral.conj(), freq),
            s=(self.n, self.n),
            norm="ortho",
        )
        if not self.no_modulation:
            s4_conj = self.local_ops.to(freq).conj()
            m = torch.einsum("mjio,bjhw->mboihw", local.to(q), context)
            op_q = torch.fft.irfft2(
                s4_conj[None, :, None] * freq[:, None],
                s=(self.n, self.n),
                norm="ortho",
            )
            spatial_adj = torch.einsum("boihw,bohw->bihw", m[0], q) + torch.einsum(
                "dboihw,bdohw->bihw", m[5:9], op_q
            )
            flux_in = torch.einsum("dboihw,bohw->bidhw", m[1:5], q)
            spec_in = (torch.fft.rfft2(flux_in, norm="ortho") * s4_conj[None, None]).sum(2)
            value = (
                value + spatial_adj + torch.fft.irfft2(spec_in, s=(self.n, self.n), norm="ortho")
            )
        return value[:, : self.state_channels], value[:, self.state_channels :]
