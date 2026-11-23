"""Public geometry masks, static volume weights, and 61-channel segment packing/unpacking."""

from __future__ import annotations

import torch

from .physics_env import BlackBoxProvider3D


def build_wet_masks_60_61(env: BlackBoxProvider3D) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct 60-channel action and 61-channel output wet masks from public geometry only."""
    geom = env.get_public_geometry()
    levels = 15
    m_u = geom["dum"].unsqueeze(0).expand(levels, -1, -1)
    m_v = geom["dvm"].unsqueeze(0).expand(levels, -1, -1)
    m_t = geom["fsm"].unsqueeze(0).expand(levels, -1, -1)
    m_s = geom["fsm"].unsqueeze(0).expand(levels, -1, -1)
    m_el = geom["fsm"].unsqueeze(0)
    mask_60 = torch.cat([m_u, m_v, m_t, m_s], dim=0)
    mask_61 = torch.cat([m_u, m_v, m_t, m_s, m_el], dim=0)
    return mask_60, mask_61


def build_obs_wet_mask_161(env: BlackBoxProvider3D, mask_60: torch.Tensor) -> torch.Tensor:
    """Build `(161, im, jm)` wet support mask for `observe_F()` using public geometry only."""
    geom = env.get_public_geometry()
    union_wet_2d = ((geom["fsm"] > 0.5) | (geom["dum"] > 0.5) | (geom["dvm"] > 0.5)).to(
        dtype=mask_60.dtype
    )
    base_mask_101 = union_wet_2d.unsqueeze(0).expand(101, -1, -1)
    return torch.cat([base_mask_101, mask_60], dim=0).contiguous()


def build_static_packing_sqrt_weights(
    env: BlackBoxProvider3D,
) -> dict[str, torch.Tensor]:
    """Build square-root static volume/area weights per field matching 5-field equal-weighted `loss_4h`."""
    geom = env.get_public_geometry()
    levels = 15
    n_wet_61 = float(
        (
            levels * geom["dum"].double().sum()
            + levels * geom["dvm"].double().sum()
            + 2 * levels * geom["fsm"].double().sum()
            + geom["fsm"].double().sum()
        )
        .clamp_min(1.0)
        .item()
    )
    field_target_sum = n_wet_61 / 5.0

    def _sqrt_norm(w_raw: torch.Tensor) -> torch.Tensor:
        w_d = w_raw.double().clamp_min(0.0)
        denom = w_d.sum().clamp_min(1e-20)
        return (w_d * (field_target_sum / denom)).sqrt().to(dtype=w_raw.dtype)

    return {
        "u": _sqrt_norm(geom["w_vol_u"]),
        "v": _sqrt_norm(geom["w_vol_v"]),
        "t": _sqrt_norm(geom["w_vol_t"]),
        "s": _sqrt_norm(geom["w_vol_t"]),
        "el": _sqrt_norm(geom["w_area_el"]),
    }


def build_volume_weights_60_61(
    env: BlackBoxProvider3D,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct static action metric W_q (60ch) and packed output metric W_o (61ch) from public geometry."""
    geom = env.get_public_geometry()
    w_u = geom["w_vol_u"].permute(2, 0, 1)
    w_v = geom["w_vol_v"].permute(2, 0, 1)
    w_t = geom["w_vol_t"].permute(2, 0, 1)

    def _norm_block(w: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        wm = (w.double() * m.double()).clamp_min(0.0)
        cnt = m.double().sum().clamp_min(1.0)
        return (wm * (cnt / wm.sum().clamp_min(1e-20))).to(dtype=w.dtype)

    levels = 15
    mu = geom["dum"].unsqueeze(0).expand(levels, -1, -1)
    mv = geom["dvm"].unsqueeze(0).expand(levels, -1, -1)
    mt = geom["fsm"].unsqueeze(0).expand(levels, -1, -1)
    mel = geom["fsm"].unsqueeze(0)

    wu_n = _norm_block(w_u, mu)
    wv_n = _norm_block(w_v, mv)
    wt_n = _norm_block(w_t, mt)
    wq_60 = torch.cat([wu_n, wv_n, wt_n, wt_n], dim=0)
    wo_61 = torch.cat([mu, mv, mt, mt, mel], dim=0).to(dtype=wq_60.dtype)
    return wq_60, wo_61


def pack_segments_to_tensor(
    env: BlackBoxProvider3D,
    segments: list[dict[str, torch.Tensor]],
) -> list[torch.Tensor]:
    """Convert 12 segment dicts into loss-scale-balanced wet-masked 61-channel tensors."""
    sw = build_static_packing_sqrt_weights(env)
    out: list[torch.Tensor] = []
    for seg in segments:
        u_s = (seg["u"] / env.loss_scales["u"]) * sw["u"]
        v_s = (seg["v"] / env.loss_scales["v"]) * sw["v"]
        t_s = (seg["t"] / env.loss_scales["t"]) * sw["t"]
        s_s = (seg["s"] / env.loss_scales["s"]) * sw["s"]
        el_s = ((seg["el"] / env.loss_scales["el"]) * sw["el"]).unsqueeze(0)
        out.append(
            torch.cat(
                [
                    u_s.permute(2, 0, 1),
                    v_s.permute(2, 0, 1),
                    t_s.permute(2, 0, 1),
                    s_s.permute(2, 0, 1),
                    el_s,
                ],
                dim=0,
            )
        )
    return out


def pack_reference_segments(
    env: BlackBoxProvider3D,
    member: int,
    day: int,
) -> list[torch.Tensor]:
    """Load the 12 reference segments for `(member, day)` as loss-scale-balanced 61-channel tensors."""
    levels = 15
    sw = build_static_packing_sqrt_weights(env)
    ref_list: list[torch.Tensor] = []
    for j in range(12):
        ref = env.dataset.load_target(member, day, (j + 1) * 1200, map_location=env.device)
        u_s = (ref["u"][:, :, :levels] / env.loss_scales["u"]) * sw["u"]
        v_s = (ref["v"][:, :, :levels] / env.loss_scales["v"]) * sw["v"]
        t_s = (ref["t"][:, :, :levels] / env.loss_scales["t"]) * sw["t"]
        s_s = (ref["s"][:, :, :levels] / env.loss_scales["s"]) * sw["s"]
        el_s = ((ref["el"] / env.loss_scales["el"]) * sw["el"]).unsqueeze(0)
        ref_list.append(
            torch.cat(
                [
                    u_s.permute(2, 0, 1),
                    v_s.permute(2, 0, 1),
                    t_s.permute(2, 0, 1),
                    s_s.permute(2, 0, 1),
                    el_s,
                ],
                dim=0,
            )
        )
    return ref_list


def build_unpacking_inv_weights(env: BlackBoxProvider3D) -> dict[str, torch.Tensor]:
    """Precompute exact inverse factors mapping packed 61-ch tensors back to physical field increments."""
    geom = env.get_public_geometry()
    sw = build_static_packing_sqrt_weights(env)
    dum3 = geom["dum"][:, :, None] > 0.5
    dvm3 = geom["dvm"][:, :, None] > 0.5
    fsm3 = geom["fsm"][:, :, None] > 0.5
    fsm2 = geom["fsm"] > 0.5

    return {
        "u": torch.where(
            dum3,
            float(env.loss_scales["u"]) / sw["u"].clamp_min(1e-20),
            torch.zeros_like(sw["u"]),
        ),
        "v": torch.where(
            dvm3,
            float(env.loss_scales["v"]) / sw["v"].clamp_min(1e-20),
            torch.zeros_like(sw["v"]),
        ),
        "t": torch.where(
            fsm3,
            float(env.loss_scales["t"]) / sw["t"].clamp_min(1e-20),
            torch.zeros_like(sw["t"]),
        ),
        "s": torch.where(
            fsm3,
            float(env.loss_scales["s"]) / sw["s"].clamp_min(1e-20),
            torch.zeros_like(sw["s"]),
        ),
        "el": torch.where(
            fsm2,
            float(env.loss_scales["el"]) / sw["el"].clamp_min(1e-20),
            torch.zeros_like(sw["el"]),
        ),
    }


def unpack_tensor_to_segments(
    env: BlackBoxProvider3D,
    packed_segments: list[torch.Tensor],
    inv_weights: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Invert `pack_segments_to_tensor` on active wet ocean cells (differentiable wrt `packed_segments`)."""
    inv_w = inv_weights if inv_weights is not None else build_unpacking_inv_weights(env)
    levels = 15
    out: list[dict[str, torch.Tensor]] = []
    for y in packed_segments:
        u_f = y[0:levels].permute(1, 2, 0) * inv_w["u"]
        v_f = y[levels : 2 * levels].permute(1, 2, 0) * inv_w["v"]
        t_f = y[2 * levels : 3 * levels].permute(1, 2, 0) * inv_w["t"]
        s_f = y[3 * levels : 4 * levels].permute(1, 2, 0) * inv_w["s"]
        el_f = y[4 * levels] * inv_w["el"]
        out.append({"u": u_f, "v": v_f, "t": t_f, "s": s_f, "el": el_f})
    return out


def pack_segment_anomalies(
    env: BlackBoxProvider3D,
    segments: list[dict[str, torch.Tensor]],
    init_fields: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Pack `(seg[f] - init_fields[f])` for 12 segments into `(12, 61, im, jm)` float32 tensor."""
    deltas = [
        {f: (seg[f].double() - init_fields[f].double()).float() for f in ("u", "v", "t", "s", "el")}
        for seg in segments
    ]
    return torch.stack(pack_segments_to_tensor(env, deltas), dim=0)


__all__ = [
    "build_obs_wet_mask_161",
    "build_static_packing_sqrt_weights",
    "build_unpacking_inv_weights",
    "build_volume_weights_60_61",
    "build_wet_masks_60_61",
    "pack_reference_segments",
    "pack_segment_anomalies",
    "pack_segments_to_tensor",
    "unpack_tensor_to_segments",
]
