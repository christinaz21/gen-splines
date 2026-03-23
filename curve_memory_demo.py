import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

IMG_SIZE = 128
NUM_CTRL = 6
NUM_SAMPLES = 200
SIGMA = 0.035           # Gaussian width for soft rasterization
LR = 0.03
STEPS_PER_VIEW = 120
SEED = 0
OUTPUT_PLOT_PATH = "curve_memory_demo_plots.png"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Utility: make image grid
# ============================================================

def make_grid(img_size: int, device: str):
    """
    Returns pixel-center grid in normalized coordinates [-1, 1].
    Shape:
        xs: [H, W]
        ys: [H, W]
    """
    coords = torch.linspace(-1.0, 1.0, img_size, device=device, dtype=DTYPE)
    ys, xs = torch.meshgrid(coords, coords, indexing="ij")
    return xs, ys


GRID_X, GRID_Y = make_grid(IMG_SIZE, DEVICE)


# ============================================================
# Curve parameterization
# ============================================================

def sample_polyline(control_points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Differentiably sample points along a polyline formed by control points.

    Args:
        control_points: [K, 2]
        num_samples: int

    Returns:
        sampled_points: [num_samples, 2]
    """
    # Segment vectors and lengths
    segs = control_points[1:] - control_points[:-1]              # [K-1, 2]
    seg_lens = torch.norm(segs, dim=-1) + 1e-8                  # [K-1]
    total_len = seg_lens.sum()

    # Sample positions uniformly along arclength
    s = torch.linspace(0.0, total_len, num_samples, device=control_points.device, dtype=control_points.dtype)

    cum = torch.cumsum(seg_lens, dim=0)                         # [K-1]
    cum_prev = torch.cat([torch.zeros(1, device=control_points.device, dtype=control_points.dtype), cum[:-1]], dim=0)

    # For each sample, figure out which segment it lies on
    seg_idx = torch.bucketize(s, cum)
    seg_idx = torch.clamp(seg_idx, max=seg_lens.shape[0] - 1)

    seg_start = control_points[seg_idx]                         # [S, 2]
    seg_vec = segs[seg_idx]                                     # [S, 2]
    seg_len = seg_lens[seg_idx]                                 # [S]
    seg_s0 = cum_prev[seg_idx]                                  # [S]

    local_t = ((s - seg_s0) / seg_len).unsqueeze(-1)            # [S, 1]
    sampled = seg_start + local_t * seg_vec                     # [S, 2]
    return sampled


def rotate_points(points: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    Rotate 2D points around origin.
    points: [N, 2]
    """
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], device=points.device, dtype=points.dtype)
    return points @ R.T


# ============================================================
# Soft differentiable rasterizer
# ============================================================

def render_curve(sampled_points: torch.Tensor, sigma: float, img_size: int) -> torch.Tensor:
    """
    Render sampled curve points into a soft image using Gaussian splats.

    Args:
        sampled_points: [N, 2], assumed in normalized coords [-1, 1]
        sigma: float
        img_size: int

    Returns:
        image: [H, W] in [0,1]
    """
    # sampled_points: [N, 2]
    px = sampled_points[:, 0].view(-1, 1, 1)   # [N,1,1]
    py = sampled_points[:, 1].view(-1, 1, 1)   # [N,1,1]

    dx2 = (GRID_X.unsqueeze(0) - px) ** 2      # [N,H,W]
    dy2 = (GRID_Y.unsqueeze(0) - py) ** 2      # [N,H,W]
    d2 = dx2 + dy2

    # Gaussian splat
    contrib = torch.exp(-d2 / (2 * sigma * sigma))  # [N,H,W]

    # Soft union of all splats
    # 1 - product(1 - alpha_i)
    image = 1.0 - torch.prod(1.0 - torch.clamp(contrib, 0.0, 0.999), dim=0)

    # Clamp to valid range
    return torch.clamp(image, 0.0, 1.0)


# ============================================================
# Ground-truth scene
# ============================================================

def make_ground_truth_control_points(device: str) -> torch.Tensor:
    """
    A simple thin-structure curve.
    """
    pts = torch.tensor(
        [
            [-0.75, -0.50],
            [-0.45,  0.15],
            [-0.10, -0.10],
            [ 0.20,  0.55],
            [ 0.48,  0.05],
            [ 0.75,  0.40],
        ],
        device=device,
        dtype=DTYPE,
    )
    return pts


def render_view(control_points: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    Rotate the curve, sample it, and render.
    """
    pts_rot = rotate_points(control_points, angle_deg)
    sampled = sample_polyline(pts_rot, NUM_SAMPLES)
    img = render_curve(sampled, SIGMA, IMG_SIZE)
    return img


# ============================================================
# Losses / metrics
# ============================================================

def image_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Simple reconstruction loss.
    """
    return F.mse_loss(pred, target)


def control_point_drift(curr_ctrl: torch.Tensor, gt_ctrl: torch.Tensor) -> torch.Tensor:
    """
    Mean Euclidean distance between control points.
    """
    return torch.norm(curr_ctrl - gt_ctrl, dim=-1).mean()


# ============================================================
# Sequential optimization experiment
# ============================================================

def optimize_single_view(init_ctrl: torch.Tensor, target_angle: float, gt_ctrl: torch.Tensor):
    """
    Fit control points to one target view.
    """
    ctrl = torch.nn.Parameter(init_ctrl.clone())

    optimizer = torch.optim.Adam([ctrl], lr=LR)
    losses = []

    target = render_view(gt_ctrl, target_angle).detach()

    for _ in range(STEPS_PER_VIEW):
        optimizer.zero_grad()

        pred = render_view(ctrl, target_angle)
        loss = image_loss(pred, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return ctrl.detach(), losses, target, render_view(ctrl.detach(), target_angle).detach()


def optimize_sequential_views(init_ctrl: torch.Tensor, gt_ctrl: torch.Tensor, view_angles):
    """
    Persistent memory experiment:
    sequentially update the SAME control points over many views.
    """
    ctrl = torch.nn.Parameter(init_ctrl.clone())
    optimizer = torch.optim.Adam([ctrl], lr=LR)

    per_step_loss = []
    per_view_final_loss = []
    drift_history = []
    revisit_error_history = []

    first_view_target = render_view(gt_ctrl, view_angles[0]).detach()

    for view_idx, angle in enumerate(view_angles):
        target = render_view(gt_ctrl, angle).detach()

        for _ in range(STEPS_PER_VIEW):
            optimizer.zero_grad()

            pred = render_view(ctrl, angle)
            loss = image_loss(pred, target)
            loss.backward()
            optimizer.step()

            per_step_loss.append(loss.item())

        with torch.no_grad():
            final_pred = render_view(ctrl, angle)
            final_loss = image_loss(final_pred, target).item()
            drift = control_point_drift(ctrl, gt_ctrl).item()

            # Revisit the first view after each update
            revisit_pred = render_view(ctrl, view_angles[0])
            revisit_error = image_loss(revisit_pred, first_view_target).item()

            per_view_final_loss.append(final_loss)
            drift_history.append(drift)
            revisit_error_history.append(revisit_error)

            print(
                f"View {view_idx:02d} | angle={angle:6.1f} | "
                f"final_loss={final_loss:.6f} | drift={drift:.4f} | revisit_err={revisit_error:.6f}"
            )

    return {
        "final_ctrl": ctrl.detach(),
        "per_step_loss": per_step_loss,
        "per_view_final_loss": per_view_final_loss,
        "drift_history": drift_history,
        "revisit_error_history": revisit_error_history,
    }


# ============================================================
# Visualization
# ============================================================

def plot_curve(ax, ctrl, title="", color="tab:blue"):
    ctrl_np = ctrl.detach().cpu().numpy()
    sampled = sample_polyline(ctrl.detach(), NUM_SAMPLES).detach().cpu().numpy()

    ax.plot(sampled[:, 0], sampled[:, 1], color=color, linewidth=2)
    ax.scatter(ctrl_np[:, 0], ctrl_np[:, 1], color="black", s=35, zorder=3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def show_image(ax, img, title=""):
    ax.imshow(img.detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


# ============================================================
# Main
# ============================================================

def main():
    gt_ctrl = make_ground_truth_control_points(DEVICE)

    # Noisy initialization = "imperfect memory"
    init_ctrl = gt_ctrl + 0.18 * torch.randn_like(gt_ctrl)

    print(f"Using device: {DEVICE}")

    # --------------------------------------------------------
    # Part 1: sanity check on a single view
    # --------------------------------------------------------
    single_angle = 25.0
    fitted_single_ctrl, single_losses, single_target, single_pred = optimize_single_view(
        init_ctrl, single_angle, gt_ctrl
    )

    # --------------------------------------------------------
    # Part 2: sequential persistent-memory experiment
    # --------------------------------------------------------
    view_angles = [-40, -20, 0, 20, 40, 20, 0, -20, -40]
    results = optimize_sequential_views(init_ctrl, gt_ctrl, view_angles)
    final_ctrl = results["final_ctrl"]

    # --------------------------------------------------------
    # Render some diagnostic images
    # --------------------------------------------------------
    gt_first = render_view(gt_ctrl, view_angles[0])
    init_first = render_view(init_ctrl, view_angles[0])
    final_first = render_view(final_ctrl, view_angles[0])
    gt_last = render_view(gt_ctrl, view_angles[-1])
    final_last = render_view(final_ctrl, view_angles[-1])

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))

    # Row 1: curves in parameter space
    ax1 = plt.subplot(3, 4, 1)
    plot_curve(ax1, gt_ctrl, "Ground Truth Curve", color="tab:green")

    ax2 = plt.subplot(3, 4, 2)
    plot_curve(ax2, init_ctrl, "Initial Memory", color="tab:red")

    ax3 = plt.subplot(3, 4, 3)
    plot_curve(ax3, fitted_single_ctrl, "After Single-View Fit", color="tab:blue")

    ax4 = plt.subplot(3, 4, 4)
    plot_curve(ax4, final_ctrl, "After Sequential Updates", color="tab:purple")

    # Row 2: rendered images
    ax5 = plt.subplot(3, 4, 5)
    show_image(ax5, gt_first, f"GT First View ({view_angles[0]}°)")

    ax6 = plt.subplot(3, 4, 6)
    show_image(ax6, init_first, "Initial Memory Render")

    ax7 = plt.subplot(3, 4, 7)
    show_image(ax7, final_first, "Revisited First View")

    ax8 = plt.subplot(3, 4, 8)
    show_image(ax8, final_last, f"Final Last View ({view_angles[-1]}°)")

    # Row 3: metrics
    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(single_losses)
    ax9.set_title("Single-View Optimization Loss")
    ax9.set_xlabel("Step")
    ax9.set_ylabel("MSE")

    ax10 = plt.subplot(3, 4, 10)
    ax10.plot(results["per_step_loss"])
    ax10.set_title("Sequential Optimization Loss")
    ax10.set_xlabel("Step")
    ax10.set_ylabel("MSE")

    ax11 = plt.subplot(3, 4, 11)
    ax11.plot(results["drift_history"], marker="o")
    ax11.set_title("Control-Point Drift Across Views")
    ax11.set_xlabel("View Index")
    ax11.set_ylabel("Mean CP Distance")

    ax12 = plt.subplot(3, 4, 12)
    ax12.plot(results["revisit_error_history"], marker="o")
    ax12.set_title("Revisitation Error to First View")
    ax12.set_xlabel("View Index")
    ax12.set_ylabel("MSE")

    plt.tight_layout()
    fig.savefig(OUTPUT_PLOT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {OUTPUT_PLOT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()