"""
hair_loader.py — Parse Cem Yuksel's .hair binary format and fit cubic B-splines.

Converts real hair strand data into the SplineField control-point representation
used by our persistent curve memory system.

Expected local dataset layout:
    data/hairmodels/straight.hair
    data/hairmodels/wCurly.hair
    data/hairmodels/wWavy.hair
    data/hairmodels/wStraight.hair
"""

import os
import struct
import numpy as np
import torch
from scipy.interpolate import splprep, splev


def load_hair_file(filepath: str) -> list:
    """
    Parse a .hair binary file into a list of strand point arrays.

    Returns:
        list of np.ndarray, each (num_points_in_strand, 3)
    """
    with open(filepath, "rb") as f:
        # --- 128-byte header ---
        signature = f.read(4)
        if signature != b"HAIR":
            raise ValueError(f"Not a HAIR file: signature={signature}")

        num_strands = struct.unpack("<I", f.read(4))[0]
        total_points = struct.unpack("<I", f.read(4))[0]
        flags = struct.unpack("<I", f.read(4))[0]

        # Defaults from header
        default_segments = struct.unpack("<I", f.read(4))[0]
        default_thickness = struct.unpack("<f", f.read(4))[0]
        default_transparency = struct.unpack("<f", f.read(4))[0]
        default_color = struct.unpack("<3f", f.read(12))

        # File info string (88 bytes)
        info = f.read(88)

        # --- Data arrays ---
        has_segments = bool(flags & 0x01)
        has_points = bool(flags & 0x02)
        has_thickness = bool(flags & 0x04)
        has_transparency = bool(flags & 0x08)
        has_color = bool(flags & 0x10)

        # Segments array
        if has_segments:
            segments = np.frombuffer(f.read(num_strands * 2), dtype=np.uint16)
        else:
            segments = np.full(num_strands, default_segments, dtype=np.uint16)

        # Points array (required)
        if has_points:
            points_flat = np.frombuffer(f.read(total_points * 3 * 4), dtype=np.float32)
            points_flat = points_flat.reshape(-1, 3)
        else:
            raise ValueError("HAIR file has no points array")

        # Thickness / transparency / color are not needed for geometry loading,
        # so we do not parse them further here.

    # Split flat points array into per-strand arrays
    strands = []
    offset = 0
    for i in range(num_strands):
        num_pts = int(segments[i]) + 1  # segments + 1 = num points
        if offset + num_pts > len(points_flat):
            break
        strand = points_flat[offset:offset + num_pts].copy()
        strands.append(strand)
        offset += num_pts

    print(f"  Loaded {len(strands)} strands from {filepath}")
    print(f"  Total points: {total_points}")
    print(
        f"  Points per strand: min={min(len(s) for s in strands)}, "
        f"max={max(len(s) for s in strands)}, "
        f"mean={np.mean([len(s) for s in strands]):.0f}"
    )

    return strands


def fit_bspline_to_strand(strand_points: np.ndarray, K: int = 8) -> np.ndarray:
    """
    Fit a cubic B-spline with K control points to a strand.

    Uses scipy's splprep for least-squares B-spline fitting, then
    extracts K evenly-spaced control points that approximate the strand.

    Args:
        strand_points: (M, 3) array of 3D points along the strand
        K: number of control points to output

    Returns:
        (K, 3) control points
    """
    M = len(strand_points)
    if M < 4:
        # Too few points — pad by repeating
        strand_points = np.vstack([strand_points] * (4 // M + 1))[:max(4, K)]
        M = len(strand_points)

    # Parameterize by arc length
    diffs = np.diff(strand_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cumlen[-1]

    if total_len < 1e-8:
        # Degenerate strand — return uniform control points
        return np.linspace(strand_points[0], strand_points[-1], K)

    u = cumlen / total_len  # normalized arc-length parameter

    # Remove duplicate parameter values (splprep requirement)
    mask = np.concatenate([[True], np.diff(u) > 1e-10])
    u_clean = u[mask]
    pts_clean = strand_points[mask]

    if len(u_clean) < 4:
        return np.linspace(strand_points[0], strand_points[-1], K)

    try:
        tck, _ = splprep(
            [pts_clean[:, 0], pts_clean[:, 1], pts_clean[:, 2]],
            u=u_clean, k=3, s=total_len * 0.001
        )
        u_eval = np.linspace(0, 1, K)
        fitted = np.array(splev(u_eval, tck)).T
        return fitted
    except Exception:
        indices = np.linspace(0, len(strand_points) - 1, K, dtype=int)
        return strand_points[indices]


def subsample_strands(
    strands: list,
    num_curves: int,
    seed: int = 42,
    min_length: int = 10,
    strategy: str = "diverse",
) -> list:
    """
    Select a subset of strands from the full hair model.

    Args:
        strands: list of strand point arrays
        num_curves: how many to select
        min_length: minimum points per strand
        strategy: "random", "diverse" (spatially spread), or "longest"

    Returns:
        list of selected strand arrays
    """
    rng = np.random.RandomState(seed)

    # Filter short strands
    valid = [s for s in strands if len(s) >= min_length]
    if len(valid) < num_curves:
        print(
            f"  Warning: only {len(valid)} strands with >={min_length} points, "
            f"requested {num_curves}"
        )
        num_curves = len(valid)

    if strategy == "random":
        indices = rng.choice(len(valid), num_curves, replace=False)
        return [valid[i] for i in indices]

    if strategy == "longest":
        lengths = [len(s) for s in valid]
        indices = np.argsort(lengths)[-num_curves:]
        return [valid[i] for i in indices]

    if strategy == "diverse":
        roots = np.array([s[0] for s in valid])
        selected = [rng.randint(len(valid))]
        for _ in range(num_curves - 1):
            dists = np.min(
                [np.linalg.norm(roots - roots[idx], axis=1) for idx in selected],
                axis=0,
            )
            selected.append(np.argmax(dists))
        return [valid[i] for i in selected]

    raise ValueError(f"Unknown strategy: {strategy}")


def hair_to_spline_field(
    strands: list,
    num_curves: int = 50,
    K: int = 8,
    seed: int = 42,
    strategy: str = "diverse",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Full pipeline: raw strands → B-spline control points tensor.

    Args:
        strands: list of strand point arrays from load_hair_file
        num_curves: number of curves to select
        K: control points per curve
        seed: random seed for strand selection
        strategy: strand selection strategy
        normalize: center and scale to unit sphere

    Returns:
        (num_curves, K, 3) tensor of control points
    """
    selected = subsample_strands(strands, num_curves, seed=seed, strategy=strategy)

    control_points = []
    for strand in selected:
        cp = fit_bspline_to_strand(strand, K=K)
        control_points.append(cp)

    cp_tensor = torch.tensor(np.stack(control_points), dtype=torch.float32)

    if normalize:
        centroid = cp_tensor.reshape(-1, 3).mean(dim=0)
        cp_tensor -= centroid

        max_dist = cp_tensor.reshape(-1, 3).norm(dim=-1).max()
        if max_dist > 1e-6:
            cp_tensor /= max_dist

    print(f"  Fitted {cp_tensor.shape[0]} curves × {K} control points")
    print(f"  CP range: [{cp_tensor.min().item():.3f}, {cp_tensor.max().item():.3f}]")

    return cp_tensor


def get_yuksel_hair_path(model_name: str = "wCurly", save_dir: str = "data") -> str:
    """
    Return the local path to a pre-downloaded Cem Yuksel hair model.

    Expected layout:
        <save_dir>/hairmodels/<model_name>.hair

    Available models expected locally:
        straight, wCurly, wWavy, wStraight

    Returns:
        Path to the .hair file
    """
    hair_dir = os.path.join(save_dir, "hairmodels")
    hair_path = os.path.join(hair_dir, f"{model_name}.hair")

    if os.path.exists(hair_path):
        print(f"  Found local hair model: {hair_path}")
        return hair_path

    raise FileNotFoundError(
        f"Could not find local hair model '{model_name}'. Expected:\n"
        f"  {hair_path}\n\n"
        f"Make sure the .hair files are present in:\n"
        f"  {hair_dir}"
    )


def download_yuksel_hair(model_name: str = "wCurly", save_dir: str = "data") -> str:
    """
    Backward-compatible wrapper.

    This project no longer downloads anything. It simply returns the expected
    local path to the requested .hair file.
    """
    return get_yuksel_hair_path(model_name=model_name, save_dir=save_dir)


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "straight"
    hair_path = get_yuksel_hair_path(model, save_dir="data")
    strands = load_hair_file(hair_path)
    cp = hair_to_spline_field(strands, num_curves=50, K=8)
    print(f"\n  Output shape: {cp.shape}")
    torch.save(cp, f"data/{model}_cp.pt")
    print(f"  Saved to data/{model}_cp.pt")
