"""
my_interp — Interpolación lineal de puntos de carril para UFLDv2.

Intenta cargar la extensión CUDA compilada. Si no está disponible,
usa una implementación vectorizada en PyTorch puro que es correcta
y razonablemente rápida para batches de entrenamiento típicos.
"""

import torch

# ── Intentar cargar el .so compilado ──────────────────────────────────────────
_cuda_ext = None
try:
    import importlib, sys, os
    # Buscar el .so compilado en site-packages o en el directorio actual
    _me = importlib.import_module('my_interp_cuda_ext')
    if hasattr(_me, 'run'):
        _cuda_ext = _me
except Exception:
    pass


# ── Implementación PyTorch vectorizada (fallback) ────────────────────────────
def _run_pytorch(points: torch.Tensor,
                 interp_loc: torch.Tensor,
                 direction: int) -> torch.Tensor:
    """
    Interpolación lineal vectorizada de puntos de carril.

    Args:
        points:      (B, L, P, 2) — pares [x, y] en espacio de píxel.
                     Los puntos sin carril tienen valor centinela -99999.
        interp_loc:  (N,) — posiciones objetivo en el eje fuente.
        direction:   0 → interpolar X dado Y  (cabeza de filas)
                     1 → interpolar Y dado X  (cabeza de columnas)

    Returns:
        (B, L, N, 2) — pares interpolados; dst=-99999 donde no hay datos.
    """
    SENTINEL = -99999.0
    # After DALI affine augmentation the sentinel coords (-99999) get transformed
    # (e.g. rotation by 6° turns x=-99999 into x≈-99417, which passes the old
    # SENTINEL+1 check). Use a looser threshold so augmented sentinels are still
    # detected.  Valid pixel x/y are always ≥ 0, so -9999 gives plenty of margin.
    VALID_THRESHOLD = -9999.0
    device = points.device
    dtype = points.dtype

    B, L, P, _ = points.shape
    N = interp_loc.shape[0]

    # Eje fuente (en el que están dados los puntos del caché)
    # y eje destino (el que hay que interpolar)
    src_axis = 1 - direction   # direction=0 → src=Y(1); direction=1 → src=X(0)
    dst_axis = direction        # direction=0 → dst=X(0); direction=1 → dst=Y(1)

    src = points[:, :, :, src_axis]   # (B, L, P)
    dst = points[:, :, :, dst_axis]   # (B, L, P)

    # Máscara de puntos válidos (dst no es centinela, incluso tras augmentación)
    valid = dst > VALID_THRESHOLD     # (B, L, P) bool

    # After augmentation the src (Y/X) axis is no longer sorted — sentinel
    # coordinates shift to large negatives (~-10000 for Y after rotation).
    # Sort ascending so searchsorted is well-defined. Sentinels sort to the
    # front; valid points (≥0) follow. This is a no-op when not augmented.
    sort_idx = torch.argsort(src, dim=2)                           # (B, L, P)
    src   = torch.gather(src,          2, sort_idx)
    dst   = torch.gather(dst,          2, sort_idx)
    valid = torch.gather(valid.long(), 2, sort_idx).bool()

    # Queries: (B, L, N)
    q = interp_loc.to(dtype=dtype, device=device).view(1, 1, N).expand(B, L, N)

    # Encontrar índices del intervalo [l, r) usando searchsorted con side='right'.
    r_idx = torch.searchsorted(src.contiguous(), q.contiguous(), right=True)  # (B, L, N)
    l_idx = r_idx - 1

    # Clamp para acceder a gather sin out-of-bounds
    r_idx_c = r_idx.clamp(0, P - 1)
    l_idx_c = l_idx.clamp(0, P - 1)

    # Recoger coordenadas de los dos extremos del intervalo
    src_l = torch.gather(src, 2, l_idx_c)                          # (B, L, N)
    src_r = torch.gather(src, 2, r_idx_c)
    dst_l = torch.gather(dst, 2, l_idx_c)
    dst_r = torch.gather(dst, 2, r_idx_c)
    valid_l = torch.gather(valid.long(), 2, l_idx_c).bool()
    valid_r = torch.gather(valid.long(), 2, r_idx_c).bool()

    # Peso de interpolación t ∈ [0, 1]
    span = (src_r - src_l).abs().clamp(min=1e-6)
    t = ((q - src_l) / span).clamp(0.0, 1.0)
    interp_val = dst_l * (1.0 - t) + dst_r * t

    # Válido si:
    #   - l_idx >= 0  (q >= src[0] del rango válido)
    #   - q <= src[:,:,-1]  (max src tras sort = máximo válido)
    #   - ambos vecinos son válidos (dst no es centinela)
    src_max = src[:, :, -1:].expand_as(q)
    in_range = (l_idx >= 0) & (q <= src_max)
    mask = in_range & valid_l & valid_r                            # (B, L, N)

    # Construir salida: centinela por defecto
    out = torch.full((B, L, N, 2), SENTINEL, dtype=dtype, device=device)
    # Posición fuente siempre conocida (es la propia query)
    out[:, :, :, src_axis] = q
    # Valor destino solo donde hay interpolación válida
    out_dst = out[:, :, :, dst_axis]
    out_dst[mask] = interp_val[mask]
    out[:, :, :, dst_axis] = out_dst

    return out


# ── API pública ───────────────────────────────────────────────────────────────
def run(points: torch.Tensor,
        interp_loc: torch.Tensor,
        direction: int) -> torch.Tensor:
    """
    Interpolación lineal de puntos de carril.

    Usa la extensión CUDA si está compilada; si no, implementación PyTorch.

    Args:
        points:      (B, L, P, 2) tensor de puntos del caché, [-99999 = sin dato]
        interp_loc:  (N,) posiciones objetivo (píxel)
        direction:   0 = interpolar X dado Y, 1 = interpolar Y dado X

    Returns:
        (B, L, N, 2) tensor interpolado
    """
    if _cuda_ext is not None:
        return _cuda_ext.run(points, interp_loc, direction)
    return _run_pytorch(points, interp_loc, direction)
