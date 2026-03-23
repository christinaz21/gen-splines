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
SIGMA = 0.035
LR = 0.03
STEPS_PER_VIEW = 120
SEED = 0
OUTPUT_PLOT_PATH = "curve_memory_demo_plots_consistency.png"

# Regularization strengths
SMOOTHNESS_WEIGHT = 0.05
CONSISTENCY_WEIGHT = 0.5

# Make baseline more fair than NUM_SAMPLES
NUM_BASELINE_POINTS = NUM_CTRL

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Utility: make image grid
# ============================================================

def make_grid(img_size: int, device: str):
    coords = torch.linspace(-1.0, 1.0, img_size, device=device, dtype=DTYPE)
    ys, xs = torch.meshgrid(coords, coords, indexing="ij")
    return xs, ys


def _init_grid_with_fallback(img_size: int, preferred_device: str):
    """
    Initialize grid tensors on CUDA when possible, otherwise gracefully fall back to CPU.
    """
    active_device = preferred_device
    try:
        grid_x, grid_y = make_grid(img_size, active_device)
    except RuntimeError as exc:
        oom_tokens = ("out of memory", "cuda error")
        if active_device == "cuda" and any(tok in str(exc).lower() for tok in oom_tokens):
            print("CUDA OOM during grid init; falling back to CPU.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            active_device = "cpu"
            grid_x, grid_y = make_grid(img_size, active_device)
        else:
            raise
    return grid_x, grid_y, active_device


GRID_X, GRID_Y, DEVICE = _init_grid_with_fallback(IMG_SIZE, DEVICE)


# ============================================================
# Curve parameterization
# ============================================================

def sample_polyline(control_points: torch.Tensor, num_samples: int) -> torch.Tensor:
    segs = control_points[1:] - control_points[:-1]
    seg_lens = torch.norm(segs, dim=-1) + 1e-8
    total_len = seg_lens.sum()

    s = torch.linspace(
        0.0, total_len,
        num_samples,
        device=control_points.device,
        dtype=control_points.dtype
    )

    cum = torch.cumsum(seg_lens, dim=0)
    cum_prev = torch.cat(
        [torch.zeros(1, device=control_points.device, dtype=control_points.dtype), cum[:-1]],
        dim=0
    )

    seg_idx = torch.bucketize(s, cum)
    seg_idx = torch.clamp(seg_idx, max=seg_lens.shape[0] - 1)

    seg_start = control_points[seg_idx]
    seg_vec = segs[seg_idx]
    seg_len = seg_lens[seg_idx]
    seg_s0 = cum_prev[seg_idx]

    local_t = ((s - seg_s0) / seg_len).unsqueeze(-1)
    sampled = seg_start + local_t * seg_vec
    return sampled


def rotate_points(points: torch.Tensor, angle_deg: float) -> torch.Tensor:
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], device=points.device, dtype=points.dtype)
    return points @ R.T


# ============================================================
# Soft differentiable rasterizer
# ============================================================

def render_curve(sampled_points: torch.Tensor, sigma: float, img_size: int) -> torch.Tensor:
    px = sampled_points[:, 0].view(-1, 1, 1)
    py = sampled_points[:, 1].view(-1, 1, 1)

    dx2 = (GRID_X.unsqueeze(0) - px) ** 2
    dy2 = (GRID_Y.unsqueeze(0) - py) ** 2
    d2 = dx2 + dy2

    contrib = torch.exp(-d2 / (2 * sigma * sigma))
    image = 1.0 - torch.prod(1.0 - torch.clamp(contrib, 0.0, 0.999), dim=0)
    return torch.clamp(image, 0.0, 1.0)


# ============================================================
# Ground-truth scene
# ============================================================

def make_ground_truth_control_points(device: str) -> torch.Tensor:
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
    pts_rot = rotate_points(control_points, angle_deg)
    sampled = sample_polyline(pts_rot, NUM_SAMPLES)
    img = render_curve(sampled, SIGMA, IMG_SIZE)
    return img


def render_point_view(points: torch.Tensor, angle_deg: float) -> torch.Tensor:
    pts_rot = rotate_points(points, angle_deg)
    img = render_curve(pts_rot, SIGMA, IMG_SIZE)
    return img


# ============================================================
# Losses / metrics
# ============================================================

def image_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def smoothness_loss(ctrl: torch.Tensor) -> torch.Tensor:
    return ((ctrl[1:] - ctrl[:-1]) ** 2).mean()


def control_point_drift(curr_ctrl: torch.Tensor, gt_ctrl: torch.Tensor) -> torch.Tensor:
    return torch.norm(curr_ctrl - gt_ctrl, dim=-1).mean()


def point_set_drift(curr_pts: torch.Tensor, gt_pts: torch.Tensor) -> torch.Tensor:
    return torch.norm(curr_pts - gt_pts, dim=-1).mean()


# ============================================================
# Structured curve-memory optimization
# ============================================================

def optimize_single_view(init_ctrl: torch.Tensor, target_angle: float, gt_ctrl: torch.Tensor):
    ctrl = torch.nn.Parameter(init_ctrl.clone())
    optimizer = torch.optim.Adam([ctrl], lr=LR)
    losses = []

    target = render_view(gt_ctrl, target_angle).detach()

    for _ in range(STEPS_PER_VIEW):
        optimizer.zero_grad()

        pred = render_view(ctrl, target_angle)
        recon = image_loss(pred, target)
        smooth = smoothness_loss(ctrl)
        loss = recon + SMOOTHNESS_WEIGHT * smooth

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return ctrl.detach(), losses, target, render_view(ctrl.detach(), target_angle).detach()


def optimize_sequential_views(init_ctrl: torch.Tensor, gt_ctrl: torch.Tensor, view_angles):
    """
    Structured curve memory with:
      - reconstruction on current view
      - smoothness regularization
      - consistency on all past views
    """
    ctrl = torch.nn.Parameter(init_ctrl.clone())
    optimizer = torch.optim.Adam([ctrl], lr=LR)

    per_step_loss = []
    per_view_final_loss = []
    drift_history = []
    revisit_error_history = []

    # Store past targets as (angle, image)
    stored_targets = []

    first_view_target = render_view(gt_ctrl, view_angles[0]).detach()

    for view_idx, angle in enumerate(view_angles):
        current_target = render_view(gt_ctrl, angle).detach()
        stored_targets.append((angle, current_target))

        for _ in range(STEPS_PER_VIEW):
            optimizer.zero_grad()

            # Current-view reconstruction
            pred = render_view(ctrl, angle)
            recon = image_loss(pred, current_target)

            # Smoothness prior
            smooth = smoothness_loss(ctrl)

            # Consistency over earlier stored views
            consistency = torch.tensor(0.0, device=ctrl.device, dtype=ctrl.dtype)
            num_past = len(stored_targets) - 1

            if num_past > 0:
                for past_angle, past_target in stored_targets[:-1]:
                    past_pred = render_view(ctrl, past_angle)
                    consistency = consistency + image_loss(past_pred, past_target)
                consistency = consistency / num_past

            loss = recon + SMOOTHNESS_WEIGHT * smooth + CONSISTENCY_WEIGHT * consistency
            loss.backward()
            optimizer.step()

            per_step_loss.append(loss.item())

        with torch.no_grad():
            final_pred = render_view(ctrl, angle)
            final_loss = image_loss(final_pred, current_target).item()
            drift = control_point_drift(ctrl, gt_ctrl).item()

            revisit_pred = render_view(ctrl, view_angles[0])
            revisit_error = image_loss(revisit_pred, first_view_target).item()

            per_view_final_loss.append(final_loss)
            drift_history.append(drift)
            revisit_error_history.append(revisit_error)

            print(
                f"[Curve+Consistency] View {view_idx:02d} | angle={angle:6.1f} | "
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
# Unstructured point baseline
# ============================================================

def optimize_sequential_point_baseline(init_points: torch.Tensor, gt_points: torch.Tensor, view_angles):
    """
    Point baseline with consistency too, so comparison is fairer.
    """
    pts = torch.nn.Parameter(init_points.clone())
    optimizer = torch.optim.Adam([pts], lr=LR)

    per_step_loss = []
    per_view_final_loss = []
    drift_history = []
    revisit_error_history = []

    stored_targets = []
    first_view_target = render_point_view(gt_points, view_angles[0]).detach()

    for view_idx, angle in enumerate(view_angles):
        current_target = render_point_view(gt_points, angle).detach()
        stored_targets.append((angle, current_target))

        for _ in range(STEPS_PER_VIEW):
            optimizer.zero_grad()

            pred = render_point_view(pts, angle)
            recon = image_loss(pred, current_target)

            consistency = torch.tensor(0.0, device=pts.device, dtype=pts.dtype)
            num_past = len(stored_targets) - 1

            if num_past > 0:
                for past_angle, past_target in stored_targets[:-1]:
                    past_pred = render_point_view(pts, past_angle)
                    consistency = consistency + image_loss(past_pred, past_target)
                consistency = consistency / num_past

            loss = recon + CONSISTENCY_WEIGHT * consistency
            loss.backward()
            optimizer.step()

            per_step_loss.append(loss.item())

        with torch.no_grad():
            final_pred = render_point_view(pts, angle)
            final_loss = image_loss(final_pred, current_target).item()
            drift = point_set_drift(pts, gt_points).item()

            revisit_pred = render_point_view(pts, view_angles[0])
            revisit_error = image_loss(revisit_pred, first_view_target).item()

            per_view_final_loss.append(final_loss)
            drift_history.append(drift)
            revisit_error_history.append(revisit_error)

            print(
                f"[Points+Consistency] View {view_idx:02d} | angle={angle:6.1f} | "
                f"final_loss={final_loss:.6f} | drift={drift:.4f} | revisit_err={revisit_error:.6f}"
            )

    return {
        "final_points": pts.detach(),
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


def plot_points(ax, points, title="", color="tab:orange"):
    pts = points.detach().cpu().numpy()
    ax.scatter(pts[:, 0], pts[:, 1], color=color, s=25)
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

    # Structured memory initialization
    init_ctrl = gt_ctrl + 0.18 * torch.randn_like(gt_ctrl)

    # Baseline points from GT curve samples
    gt_points = sample_polyline(gt_ctrl, NUM_BASELINE_POINTS).detach()
    init_points = gt_points + 0.18 * torch.randn_like(gt_points)

    print(f"Using device: {DEVICE}")
    print(f"SMOOTHNESS_WEIGHT = {SMOOTHNESS_WEIGHT}")
    print(f"CONSISTENCY_WEIGHT = {CONSISTENCY_WEIGHT}")
    print(f"NUM_BASELINE_POINTS = {NUM_BASELINE_POINTS}")

    # --------------------------------------------------------
    # Part 1: sanity check on a single view
    # --------------------------------------------------------
    single_angle = 25.0
    fitted_single_ctrl, single_losses, single_target, single_pred = optimize_single_view(
        init_ctrl, single_angle, gt_ctrl
    )

    # --------------------------------------------------------
    # Part 2: sequential curve memory with consistency
    # --------------------------------------------------------
    view_angles = [-40, -20, 0, 20, 40, 20, 0, -20, -40]
    curve_results = optimize_sequential_views(init_ctrl, gt_ctrl, view_angles)
    final_ctrl = curve_results["final_ctrl"]

    # --------------------------------------------------------
    # Part 3: sequential point baseline with consistency
    # --------------------------------------------------------
    point_results = optimize_sequential_point_baseline(init_points, gt_points, view_angles)
    final_points = point_results["final_points"]

    # --------------------------------------------------------
    # Render diagnostic images
    # --------------------------------------------------------
    gt_first = render_view(gt_ctrl, view_angles[0])
    init_first = render_view(init_ctrl, view_angles[0])
    final_first = render_view(final_ctrl, view_angles[0])

    point_init_first = render_point_view(init_points, view_angles[0])
    point_final_first = render_point_view(final_points, view_angles[0])

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    fig = plt.figure(figsize=(18, 14))

    ax1 = plt.subplot(4, 4, 1)
    plot_curve(ax1, gt_ctrl, "Ground Truth Curve", color="tab:green")

    ax2 = plt.subplot(4, 4, 2)
    plot_curve(ax2, init_ctrl, "Initial Curve Memory", color="tab:red")

    ax3 = plt.subplot(4, 4, 3)
    plot_curve(ax3, fitted_single_ctrl, "Single-View Fit (Curve)", color="tab:blue")

    ax4 = plt.subplot(4, 4, 4)
    plot_curve(ax4, final_ctrl, "Final Curve Memory", color="tab:purple")

    ax5 = plt.subplot(4, 4, 5)
    plot_points(ax5, gt_points, "GT Unstructured Points", color="tab:green")

    ax6 = plt.subplot(4, 4, 6)
    plot_points(ax6, init_points, "Initial Point Baseline", color="tab:red")

    ax7 = plt.subplot(4, 4, 7)
    plot_points(ax7, final_points, "Final Point Baseline", color="tab:orange")

    ax8 = plt.subplot(4, 4, 8)
    show_image(ax8, gt_first, f"GT First View ({view_angles[0]}°)")

    ax9 = plt.subplot(4, 4, 9)
    show_image(ax9, init_first, "Initial Curve Render")

    ax10 = plt.subplot(4, 4, 10)
    show_image(ax10, final_first, "Revisited Curve Render")

    ax11 = plt.subplot(4, 4, 11)
    show_image(ax11, point_init_first, "Initial Point Render")

    ax12 = plt.subplot(4, 4, 12)
    show_image(ax12, point_final_first, "Revisited Point Render")

    ax13 = plt.subplot(4, 4, 13)
    ax13.plot(single_losses)
    ax13.set_title("Single-View Loss (Curve)")
    ax13.set_xlabel("Step")
    ax13.set_ylabel("Loss")

    ax14 = plt.subplot(4, 4, 14)
    ax14.plot(curve_results["drift_history"], marker="o", label="Curve + Smooth + Consistency")
    ax14.plot(point_results["drift_history"], marker="o", label="Points + Consistency")
    ax14.set_title("Drift Across Views")
    ax14.set_xlabel("View Index")
    ax14.set_ylabel("Mean Distance")
    ax14.legend()

    ax15 = plt.subplot(4, 4, 15)
    ax15.plot(curve_results["revisit_error_history"], marker="o", label="Curve + Smooth + Consistency")
    ax15.plot(point_results["revisit_error_history"], marker="o", label="Points + Consistency")
    ax15.set_title("Revisitation Error")
    ax15.set_xlabel("View Index")
    ax15.set_ylabel("MSE")
    ax15.legend()

    ax16 = plt.subplot(4, 4, 16)
    ax16.plot(curve_results["per_step_loss"], label="Curve + Smooth + Consistency")
    ax16.plot(point_results["per_step_loss"], label="Points + Consistency")
    ax16.set_title("Sequential Optimization Loss")
    ax16.set_xlabel("Step")
    ax16.set_ylabel("Loss")
    ax16.legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_PLOT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {OUTPUT_PLOT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()