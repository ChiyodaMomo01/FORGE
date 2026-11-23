"""Float64 high-resolution physical diagnostics and radial power spectra."""

import torch


def serial(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    return value


def spectrum(x, length):
    """Radial shell sums (cycles/domain); mean removed, Parseval normalization."""
    n = x.shape[-1]
    k = torch.fft.fftfreq(n, device=x.device, dtype=x.dtype) * n
    kr = (k[:, None].square() + k[None, :].square()).sqrt()
    power = torch.fft.fft2(x - x.mean((-2, -1), keepdim=True)).abs().square() / n**4
    power = power.sum(tuple(range(power.ndim - 2)))
    edges = torch.linspace(0, n / 2, 17, device=x.device, dtype=x.dtype)
    bins = torch.bucketize(kr.flatten(), edges[1:], right=True).clamp(max=16)
    shells = torch.zeros(17, device=x.device, dtype=x.dtype).scatter_add_(0, bins, power.flatten())
    denominator = (power * kr).sum()
    scale = length * power.sum() / denominator if denominator > 0 else x.new_tensor(0.0)
    return {
        "spectrum": serial(shells),
        "spectrum_edges": serial(edges),
        "spectral_length": serial(scale),
    }


def diagnose(case, state):
    x = case.fields(state)
    if x.dtype != torch.float64 or (
        x.device.type != "cpu" and case.name not in ("kolmogorov_ns", "mhd2d")
    ):
        raise ValueError("reference diagnostics require float64 fields")
    if not bool(x.isfinite().all()):
        raise FloatingPointError("nonfinite HR state")
    out = {
        "time": serial(state.get("time", 0.0)),
        "mean": serial(x.mean((-2, -1))),
        "variance": serial(x.var((-2, -1), unbiased=False)),
        "minimum": serial(x.amin((-2, -1))),
        "maximum": serial(x.amax((-2, -1))),
    }
    out.update({k: serial(v) for k, v in case.diagnostics(state).items()})
    out.update(spectrum(x[:, -1:], case.config.length))
    if case.name == "gray_scott":
        out["active_fraction_v_gt_0.1"] = serial((x[:, 1] > 0.1).double().mean())
        out["reaction_rms"] = serial(case._reaction(x, None).square().mean().sqrt())
        if x.min() < -1e-10:
            raise FloatingPointError("negative HR Gray-Scott concentration")
    return out
