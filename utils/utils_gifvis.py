import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# Visually distinct RGB palette (values in [0, 1]) — matches plot color order
_VIEW_COLORS = np.array([
    [0.122, 0.467, 0.706],   # blue
    [1.000, 0.498, 0.055],   # orange
    [0.173, 0.627, 0.173],   # green
    [0.839, 0.153, 0.157],   # red
    [0.580, 0.404, 0.741],   # purple
    [0.549, 0.337, 0.294],   # brown
    [0.890, 0.467, 0.761],   # pink
    [0.498, 0.498, 0.498],   # gray
    [0.737, 0.741, 0.133],   # olive
    [0.090, 0.745, 0.812],   # cyan
], dtype=np.float64)


# ── private helpers ───────────────────────────────────────────────────────────

def _detect_view_clouds(folder):
    """Return paths to cloud_{i}.pcd files sorted by integer index i."""
    candidates = glob.glob(os.path.join(folder, "cloud_*.pcd"))
    view_clouds = []
    for path in candidates:
        name   = os.path.splitext(os.path.basename(path))[0]
        suffix = name[len("cloud_"):]
        if suffix.isdigit():
            view_clouds.append((int(suffix), path))
    view_clouds.sort()
    return [p for _, p in view_clouds]


def _build_colored_accumulation(cloud_paths, dup_threshold=0.001):
    """
    Return a list of colored PointClouds, one per view.

    Each entry is the accumulated cloud up to and including view i.
    Points introduced at view i are colored with color_i.
    Existing points that are within dup_threshold of a new point are
    recolored to color_i (newer view wins).

    Uses scipy.cKDTree for vectorized nearest-neighbor queries when
    available; falls back to the Open3D KD-tree otherwise.
    """
    n       = len(cloud_paths)
    palette = np.tile(_VIEW_COLORS, (n // len(_VIEW_COLORS) + 1, 1))[:n]

    acc_pts = np.empty((0, 3), dtype=np.float64)
    acc_col = np.empty((0, 3), dtype=np.float64)
    frames  = []

    for i, path in enumerate(cloud_paths):
        raw = o3d.io.read_point_cloud(
            path, remove_nan_points=True, remove_infinite_points=True
        )
        pts   = np.asarray(raw.points, dtype=np.float64)
        col_i = palette[i]

        if len(pts) == 0:
            pcd = o3d.geometry.PointCloud()
            if len(acc_pts):
                pcd.points = o3d.utility.Vector3dVector(acc_pts.copy())
                pcd.colors = o3d.utility.Vector3dVector(acc_col.copy())
            frames.append(pcd)
            continue

        if len(acc_pts) == 0:
            acc_pts = pts.copy()
            acc_col = np.tile(col_i, (len(pts), 1))
        else:
            if _HAS_SCIPY:
                tree_new  = cKDTree(pts)
                tree_acc  = cKDTree(acc_pts)

                # recolor existing points that overlap with the new cloud
                d_exist, _ = tree_new.query(acc_pts)
                acc_col[d_exist < dup_threshold] = col_i

                # keep only truly new points
                d_new, _   = tree_acc.query(pts)
                truly_new  = pts[d_new >= dup_threshold]
            else:
                # Open3D KD-tree fallback (point-by-point, slower)
                acc_pcd        = o3d.geometry.PointCloud()
                acc_pcd.points = o3d.utility.Vector3dVector(acc_pts)
                tree      = o3d.geometry.KDTreeFlann(acc_pcd)
                new_mask  = np.ones(len(pts), dtype=bool)
                for j, pt in enumerate(pts):
                    k, idx, _ = tree.search_radius_vector_3d(pt, dup_threshold)
                    if k > 0:
                        new_mask[j]      = False
                        acc_col[idx[0]]  = col_i
                truly_new = pts[new_mask]

            if len(truly_new):
                acc_pts = np.vstack([acc_pts, truly_new])
                acc_col = np.vstack([acc_col, np.tile(col_i, (len(truly_new), 1))])

        pcd        = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(acc_pts.copy())
        pcd.colors = o3d.utility.Vector3dVector(acc_col.copy())
        frames.append(pcd)

    return frames


def _make_renderer(width, height, bg_color, point_size):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(np.array(bg_color, dtype=np.float32))
    mat           = o3d.visualization.rendering.MaterialRecord()
    mat.shader    = "defaultUnlit"
    mat.point_size = float(point_size)
    return renderer, mat


def _orbit_eye(center, radius, elevation_deg, azimuth_deg):
    """Camera position on a sphere of given radius around center."""
    az  = np.radians(azimuth_deg)
    el  = np.radians(elevation_deg)
    return center + radius * np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])


def _render_frame(renderer, mat, pcd, fov, center, radius, elevation_deg, azimuth_deg):
    """
    Place pcd in the renderer, position the camera, and return an (H, W, 3)
    uint8 numpy array.
    """
    try:
        renderer.scene.remove_geometry("pcd")
    except Exception:
        pass
    renderer.scene.add_geometry("pcd", pcd, mat)

    eye = _orbit_eye(center, radius, elevation_deg, azimuth_deg)
    up  = np.array([0.0, 0.0, 1.0])
    renderer.setup_camera(fov, center.tolist(), eye.tolist(), up.tolist())

    img = np.asarray(renderer.render_to_image())
    return img[:, :, :3]  # drop alpha channel


def _text_overlay(frame_rgb, text):
    """Burn a text label into the top-left corner (requires PIL)."""
    if not _HAS_PIL:
        return frame_rgb
    img  = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    # Draw a semi-transparent shadow first for readability
    draw.text((11, 11), text, fill=(200, 200, 200))
    draw.text((10, 10), text, fill=(30,  30,  30))
    return np.asarray(img)


def _save_gif(frames, durations_ms, output_path):
    """
    Save frames as a GIF with per-frame durations.

    Parameters
    ----------
    frames : list of np.ndarray  (H, W, 3) uint8
    durations_ms : list of int   milliseconds per frame (same length as frames)
    output_path : str
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    uint8_frames = [f.astype(np.uint8) for f in frames]

    if _HAS_IMAGEIO:
        try:
            import imageio.v3 as iio3
            iio3.imwrite(output_path, uint8_frames, plugin="pillow",
                         duration=durations_ms, loop=0)
        except (AttributeError, ImportError):
            # imageio v2 does not support per-frame duration; use median
            median_fps = int(round(1000 / (sum(durations_ms) / len(durations_ms))))
            imageio.mimsave(output_path, uint8_frames, fps=max(median_fps, 1))
    elif _HAS_PIL:
        imgs = [Image.fromarray(f) for f in uint8_frames]
        imgs[0].save(output_path, save_all=True, append_images=imgs[1:],
                     loop=0, duration=durations_ms)
    else:
        raise ImportError(
            "imageio or PIL is required. Install with:\n"
            "    pip install imageio[pillow]"
        )


# ── public API ────────────────────────────────────────────────────────────────

def create_reconstruction_gif(
    dir_carpeta,
    itera,
    output_path,
    n_views=None,
    rotation_frames=72,
    accum_ms=800,
    rotation_fps=18,
    width=800,
    height=600,
    fov=60.0,
    point_size=3.0,
    dup_threshold=0.001,
    elevation=25.0,
    start_azimuth=45.0,
    bg_color=(1.0, 1.0, 1.0, 1.0),
    show_labels=True,
    show_gt=False,
    gt_color=(0.0, 0.0, 0.0),
    zoom=1.0,
):
    """
    Create a GIF showing progressive 3D reconstruction followed by a 360° rotation.

    The function reads cloud_{i}.pcd files from ``dir_carpeta + itera``, builds
    a frame-by-frame accumulation with per-view coloring, and appends a smooth
    orbital rotation of the final colored cloud.

    Parameters
    ----------
    dir_carpeta : str
        Base point cloud directory (e.g. '/data/obj/Point_cloud/NBVNet/').
        Follows the same convention as utils_metrics.py.
    itera : str
        Iteration subfolder with trailing slash (e.g. 'iter1/').
        Full path = dir_carpeta + itera + 'cloud_N.pcd'.
    output_path : str
        Output GIF file path.
    n_views : int, optional
        Max number of views to include. Auto-detects all cloud_{i}.pcd if None.
    rotation_frames : int
        Frames for the final 360° rotation (default 72 = 5° per frame).
    accum_ms : int
        Display duration in milliseconds for each accumulation frame.
        Increase this value to slow down the accumulation phase (default 800 ms).
    rotation_fps : int
        Playback speed for the 360° rotation phase in frames per second.
    width, height : int
        Render resolution in pixels.
    fov : float
        Camera vertical field of view in degrees.
    point_size : float
        Rendered point size in pixels.
    dup_threshold : float
        Distance below which two points are treated as the same location
        for color-update purposes. Should match the voxel size used during
        reconstruction (default 0.001 m).
    elevation : float
        Camera elevation above the XY plane in degrees (0 = equator, 90 = top).
        Keep below 80° to avoid gimbal lock with the Z-up vector.
    start_azimuth : float
        Camera azimuth for accumulation frames in degrees; the rotation phase
        starts from this angle and goes +360°.
    bg_color : tuple
        RGBA background color with values in [0, 1].
    show_labels : bool
        Overlay 'View N/M' text on accumulation frames (requires PIL).
    show_gt : bool
        If True, render the ground-truth cloud (cloud_gt.pcd) in black as a
        fixed background layer visible in every frame.
    gt_color : tuple
        RGB color for the GT cloud, values in [0, 1]. Default is black.

    Notes
    -----
    Rendering requires Open3D's OffscreenRenderer, which on Linux needs either
    an X display, EGL, or OSMesa.  With the ``o3d`` conda environment and
    Mesa drivers the default headless path usually works.

    Examples
    --------
    >>> create_reconstruction_gif(
    ...     dir_carpeta = '/data/nefertiti/Point_cloud/NBVNet/',
    ...     itera       = 'iter1/',
    ...     output_path = 'nefertiti_nbvnet.gif',
    ... )
    """
    folder = dir_carpeta + itera

    cloud_paths = _detect_view_clouds(folder)
    if n_views is not None:
        cloud_paths = cloud_paths[:n_views]
    if not cloud_paths:
        raise FileNotFoundError(f"No cloud_N.pcd files found in: {folder}")

    print(f"[GIF] {len(cloud_paths)} view cloud(s) detected in {folder}")

    # ── Phase 1: build per-frame colored accumulation ──────────────────────
    print("[GIF] Building colored accumulation (this may take a moment)...")
    acc_frames = _build_colored_accumulation(cloud_paths, dup_threshold)

    # ── Camera reference ───────────────────────────────────────────────────
    # Prefer the ground-truth cloud for a stable center across all frames.
    gt_path = os.path.join(folder, "cloud_gt.pcd")
    if os.path.exists(gt_path):
        ref_pcd = o3d.io.read_point_cloud(gt_path,
                                          remove_nan_points=True,
                                          remove_infinite_points=True)
    else:
        ref_pcd = acc_frames[-1]

    center = np.asarray(ref_pcd.get_center())
    extent = np.max(ref_pcd.get_axis_aligned_bounding_box().get_extent())
    radius = extent * 1.8 / zoom

    # ── Setup renderer ─────────────────────────────────────────────────────
    renderer, mat = _make_renderer(width, height, bg_color, point_size)
    gif_frames    = []
    total_views   = len(acc_frames)

    # Add GT cloud once — it stays in the scene for all frames
    if show_gt and os.path.exists(gt_path):
        gt_pcd = o3d.io.read_point_cloud(gt_path,
                                         remove_nan_points=True,
                                         remove_infinite_points=True)
        n_gt   = len(gt_pcd.points)
        gt_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(gt_color, (n_gt, 1))
        )
        renderer.scene.add_geometry("gt", gt_pcd, mat)

    # ── Phase 2: accumulation animation ───────────────────────────────────
    print("[GIF] Rendering accumulation frames...")
    n_accum_rendered = 0
    for i, pcd in enumerate(acc_frames):
        if len(pcd.points) == 0:
            continue
        img = _render_frame(renderer, mat, pcd, fov, center, radius,
                             elevation, start_azimuth)
        if show_labels:
            img = _text_overlay(img, f"View {i + 1}/{total_views}")
        gif_frames.append(img)
        n_accum_rendered += 1

    # ── Phase 3: 360° orbital rotation ────────────────────────────────────
    print("[GIF] Rendering rotation frames...")
    final_pcd = acc_frames[-1]
    for az in np.linspace(start_azimuth, start_azimuth + 360.0,
                           rotation_frames, endpoint=False):
        img = _render_frame(renderer, mat, final_pcd, fov, center,
                             radius, elevation, az)
        gif_frames.append(img)

    # ── Build per-frame duration list ──────────────────────────────────────
    rotation_ms  = int(round(1000 / rotation_fps))
    durations_ms = ([int(accum_ms)] * n_accum_rendered
                    + [rotation_ms]  * rotation_frames)

    # ── Save ───────────────────────────────────────────────────────────────
    print(f"[GIF] Saving {len(gif_frames)} frames → {output_path}")
    _save_gif(gif_frames, durations_ms, output_path)
    print("[GIF] Done.")

    del renderer  # release GL context


def merge_gifs_side_by_side(gif_paths, output_path, labels=None,
                             target_height=None, gap=4,
                             gap_color=(255, 255, 255)):
    """
    Concatenate GIFs horizontally and save as a single GIF.

    All GIFs are resized to the same height (tallest by default).
    If the GIFs have different frame counts the last frame is held.
    Frame duration at each step is the maximum among all inputs so
    no GIF appears to skip ahead.

    Parameters
    ----------
    gif_paths : list of str
        Paths to the input GIFs (any number).
    output_path : str
        Path for the merged output GIF.
    labels : list of str, optional
        Short text labels drawn at the top of each panel (requires PIL).
    target_height : int, optional
        Resize every panel to this height in pixels.
        Defaults to the height of the tallest input GIF.
    gap : int
        Width of the vertical separator between panels in pixels.
    gap_color : tuple
        RGB colour of the separator (default white).

    Examples
    --------
    >>> merge_gifs_side_by_side(
    ...     ['nbvnet.gif', 'pcnbv.gif', 'random.gif'],
    ...     output_path = 'comparison.gif',
    ...     labels      = ['NBV-Net', 'PC-NBV', 'Random'],
    ... )
    """
    if not _HAS_PIL:
        raise ImportError("PIL is required. Install with: pip install pillow")

    def _load_gif(path):
        """Return (list[RGBA Image], list[int ms]) for every frame."""
        gif    = Image.open(path)
        frames, durations = [], []
        try:
            while True:
                frames.append(gif.copy().convert("RGBA"))
                durations.append(gif.info.get("duration", 100))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
        return frames, durations

    all_frames, all_durations = zip(*[_load_gif(p) for p in gif_paths])
    n_gifs    = len(gif_paths)
    n_frames  = max(len(f) for f in all_frames)

    # Determine common height
    heights = [f[0].height for f in all_frames]
    h       = target_height or max(heights)

    def _resize_to_height(img, new_h):
        if img.height == new_h:
            return img
        ratio = new_h / img.height
        return img.resize((int(img.width * ratio), new_h), Image.LANCZOS)

    # Resize first frames to get panel widths
    sample_panels = [_resize_to_height(f[0], h) for f in all_frames]
    widths        = [p.width for p in sample_panels]
    total_w       = sum(widths) + gap * (n_gifs - 1)

    label_h = 24 if labels else 0
    canvas_h = h + label_h

    merged_frames, merged_durations = [], []

    for fi in range(n_frames):
        canvas = Image.new("RGBA", (total_w, canvas_h), (255, 255, 255, 255))
        x      = 0
        dur_i  = 0

        for gi in range(n_gifs):
            # Hold last frame if this GIF is shorter
            frame_idx = min(fi, len(all_frames[gi]) - 1)
            panel     = _resize_to_height(all_frames[gi][frame_idx], h)
            dur_i     = max(dur_i, all_durations[gi][frame_idx])

            # Draw label above panel
            if labels and gi < len(labels):
                from PIL import ImageDraw
                draw = ImageDraw.Draw(canvas)
                draw.text((x + panel.width // 2 - len(labels[gi]) * 3, 4),
                          labels[gi], fill=(30, 30, 30))

            canvas.paste(panel, (x, label_h))
            x += panel.width

            # Draw gap (skip after last panel)
            if gi < n_gifs - 1:
                gap_img = Image.new("RGBA", (gap, canvas_h), gap_color + (255,))
                canvas.paste(gap_img, (x, 0))
                x += gap

        merged_frames.append(np.asarray(canvas.convert("RGB")))
        merged_durations.append(dur_i)

    _save_gif(merged_frames, merged_durations, output_path)
    print(f"[merge_gifs] Saved → {output_path}")


def reconstruction_snapshots(
    dir_carpeta,
    itera,
    output_path,
    views=(0, 4, 8),
    labels=None,
    width=800,
    height=600,
    fov=60.0,
    point_size=3.0,
    dup_threshold=0.001,
    elevation=25.0,
    azimuth=45.0,
    bg_color=(1.0, 1.0, 1.0, 1.0),
    gap=8,
    gap_color=(240, 240, 240),
    show_gt=False,
    gt_color=(0.0, 0.0, 0.0),
    save_eps=True,
    zoom=1.0,
):
    """
    Render the accumulated point cloud at selected view indices, then
    place the snapshots side by side and save as PNG (and optionally EPS).

    Parameters
    ----------
    dir_carpeta : str
        Base point cloud directory (same convention as utils_metrics.py).
    itera : str
        Iteration subfolder with trailing slash.
    output_path : str
        Output file path *without* extension.  The function writes
        ``output_path.png`` and, if save_eps=True, ``output_path.eps``.
    views : tuple/list of int
        0-based indices of the view clouds to capture
        (0 = cloud_0.pcd, 4 = cloud_4.pcd, …).
        Default (0, 4, 8) corresponds to views 1, 5 and 9.
    labels : list of str, optional
        One label per panel.  Defaults to 'View N' (1-based).
    width, height : int
        Render resolution per panel in pixels.
    fov : float
        Camera vertical field of view in degrees.
    point_size : float
        Rendered point size in pixels.
    dup_threshold : float
        Distance threshold for duplicate-point detection.
    elevation : float
        Camera elevation above XY plane in degrees.
    azimuth : float
        Camera azimuth in degrees (the fixed "initial perspective").
    bg_color : tuple
        RGBA background color, values in [0, 1].
    gap : int
        Pixel width of the separator between panels.
    gap_color : tuple
        RGB color of the separator.
    show_gt : bool
        Overlay the ground-truth cloud in gt_color on every panel.
    gt_color : tuple
        RGB color for the GT cloud (values in [0, 1]).
    save_eps : bool
        Also save an EPS file alongside the PNG.

    Examples
    --------
    >>> reconstruction_snapshots(
    ...     dir_carpeta = '/data/nefertiti/Point_cloud/NBVNet/',
    ...     itera       = 'iter1/',
    ...     output_path = 'snapshots/nefertiti_nbvnet',
    ...     views       = (0, 4, 8),
    ...     labels      = ['View 1', 'View 5', 'View 9'],
    ... )
    """
    if not _HAS_PIL:
        raise ImportError("PIL is required. Install with: pip install pillow")

    folder      = dir_carpeta + itera
    cloud_paths = _detect_view_clouds(folder)
    if not cloud_paths:
        raise FileNotFoundError(f"No cloud_N.pcd files found in: {folder}")

    max_view = max(views)
    if max_view >= len(cloud_paths):
        raise IndexError(
            f"Requested view index {max_view} but only "
            f"{len(cloud_paths)} clouds found in {folder}"
        )

    if labels is None:
        labels = [f"View {v + 1}" for v in views]

    # Build accumulation only up to the highest requested view
    print("[snapshots] Building colored accumulation...")
    acc_frames = _build_colored_accumulation(
        cloud_paths[: max_view + 1], dup_threshold
    )

    # Camera reference
    gt_path = os.path.join(folder, "cloud_gt.pcd")
    if os.path.exists(gt_path):
        ref_pcd = o3d.io.read_point_cloud(gt_path,
                                          remove_nan_points=True,
                                          remove_infinite_points=True)
    else:
        ref_pcd = acc_frames[-1]

    center = np.asarray(ref_pcd.get_center())
    extent = np.max(ref_pcd.get_axis_aligned_bounding_box().get_extent())
    radius = extent * 1.8 / zoom

    renderer, mat = _make_renderer(width, height, bg_color, point_size)

    if show_gt and os.path.exists(gt_path):
        gt_pcd = o3d.io.read_point_cloud(gt_path,
                                         remove_nan_points=True,
                                         remove_infinite_points=True)
        gt_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(gt_color, (len(gt_pcd.points), 1))
        )
        renderer.scene.add_geometry("gt", gt_pcd, mat)

    # Render each requested view
    print("[snapshots] Rendering panels...")
    panels = []
    for v in views:
        img = _render_frame(renderer, mat, acc_frames[v],
                             fov, center, radius, elevation, azimuth)
        panels.append(img)

    del renderer

    # ── Compose side-by-side image ─────────────────────────────────────────
    label_h   = 28 if labels else 0
    total_w   = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas_h  = height + label_h
    canvas    = Image.new("RGB", (total_w, canvas_h), (255, 255, 255))
    draw      = ImageDraw.Draw(canvas)

    x = 0
    for i, (panel, label) in enumerate(zip(panels, labels)):
        pil_panel = Image.fromarray(panel)
        canvas.paste(pil_panel, (x, label_h))

        # Centered label
        text_x = x + panel.shape[1] // 2 - len(label) * 4
        draw.text((text_x + 1, 7), label, fill=(180, 180, 180))  # shadow
        draw.text((text_x,     6), label, fill=(30,  30,  30))

        x += panel.shape[1]
        if i < len(panels) - 1:
            gap_rect = Image.new("RGB", (gap, canvas_h), gap_color)
            canvas.paste(gap_rect, (x, 0))
            x += gap

    # ── Save PNG ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path + ".png")),
                exist_ok=True)
    png_path = output_path + ".png"
    canvas.save(png_path)
    print(f"[snapshots] PNG saved → {png_path}")

    # ── Save EPS ───────────────────────────────────────────────────────────
    if save_eps:
        eps_path = output_path + ".eps"
        # PIL writes EPS natively; convert to RGB first (no alpha)
        canvas.convert("RGB").save(eps_path, format="EPS")
        print(f"[snapshots] EPS saved → {eps_path}")


def reconstruction_comparison_grid(
    methods,
    output_path,
    views=(0, 4, 8),
    col_labels=None,
    width=600,
    height=500,
    fov=60.0,
    point_size=3.0,
    dup_threshold=0.001,
    elevation=25.0,
    azimuth=45.0,
    bg_color=(1.0, 1.0, 1.0, 1.0),
    show_gt=False,
    gt_color=(0.0, 0.0, 0.0),
    zoom=1.0,
    dpi=150,
    coverage_fontsize=11,
    method_fontsize=11,
    col_label_fontsize=11,
    panel_labels=True,
    panel_label_fontsize=12,
    panel_label_loc='lower left',
):
    """
    Render a method × view grid figure, matching the layout of comparison
    figures in NBV papers (method label on the left, coverage % above each
    panel, column labels below, dashed separators between rows).

    Parameters
    ----------
    methods : list of dict
        One entry per row.  Each dict must contain:
          - 'name'        : str   — row label (e.g. 'NBV-Net')
          - 'dir_carpeta' : str   — base point cloud directory
          - 'itera'       : str   — iteration subfolder
        Optional keys:
          - 'coverages'   : list of float — coverage % for each column
            (e.g. [22.5, 52.7, 76.5]).  Omit to skip percentage labels.
    output_path : str
        Output path without extension.  Writes .png and .eps.
    views : tuple of int
        0-based view indices to capture per method (one column each).
    col_labels : list of str, optional
        Labels for the bottom of each column.
        Defaults to ['(a) View 1', '(b) View 5', ...].
    width, height : int
        Render resolution per panel in pixels.
    fov, point_size, dup_threshold, elevation, azimuth : float
        Passed to the renderer (same as reconstruction_snapshots).
    bg_color : tuple   RGBA background for renders.
    show_gt : bool     Overlay GT cloud in gt_color on every panel.
    gt_color : tuple   RGB for GT cloud.
    zoom : float       Camera zoom (>1 = closer).
    dpi : int          Figure DPI for saved files.
    coverage_fontsize, method_fontsize, col_label_fontsize : int
        Font sizes for the three text types.
    panel_labels : bool
        If True, stamp each individual panel with a unique sub-index
        (a), b), c), ... in row-major reading order, top-left to
        bottom-right across the whole grid.
    panel_label_fontsize : int
        Font size for the panel sub-index labels.
    panel_label_loc : str
        Corner of each panel to place the sub-index label.  One of
        'lower left', 'lower right', 'upper left', 'upper right'.

    Examples
    --------
    >>> reconstruction_comparison_grid(
    ...     methods = [
    ...         {'name': 'NBV-Net',  'dir_carpeta': '/data/obj/Point_cloud/NBVNet/',
    ...          'itera': 'iter1/', 'coverages': [22.5, 52.7, 76.5]},
    ...         {'name': 'PC-NBV',   'dir_carpeta': '/data/obj/Point_cloud/PCNBV/',
    ...          'itera': 'iter1/', 'coverages': [22.5, 60.5, 86.6]},
    ...     ],
    ...     output_path = 'figures/comparison_dragon',
    ...     views       = (0, 4, 8),
    ...     col_labels  = ['(a) 1 round', '(b) 2 rounds', '(c) 3 rounds'],
    ... )
    """
    n_rows = len(methods)
    n_cols = len(views)

    if col_labels is None:
        letters = 'abcdefghij'
        col_labels = [f'({letters[i]}) View {v + 1}'
                      for i, v in enumerate(views)]

    # ── Render all panels ──────────────────────────────────────────────────
    all_panels = []   # list of rows; each row = list of (H,W,3) arrays

    for m in methods:
        folder      = m['dir_carpeta'] + m['itera']
        cloud_paths = _detect_view_clouds(folder)
        if not cloud_paths:
            raise FileNotFoundError(f"No cloud_N.pcd in {folder}")

        max_view = max(views)
        if max_view >= len(cloud_paths):
            raise IndexError(
                f"View {max_view} requested but only "
                f"{len(cloud_paths)} clouds in {folder}"
            )

        print(f"[grid] Building accumulation for '{m['name']}'...")
        acc_frames = _build_colored_accumulation(
            cloud_paths[: max_view + 1], dup_threshold
        )

        gt_path = os.path.join(folder, "cloud_gt.pcd")
        ref_pcd = (o3d.io.read_point_cloud(gt_path,
                                            remove_nan_points=True,
                                            remove_infinite_points=True)
                   if os.path.exists(gt_path) else acc_frames[-1])

        center = np.asarray(ref_pcd.get_center())
        extent = np.max(ref_pcd.get_axis_aligned_bounding_box().get_extent())
        radius = extent * 1.8 / zoom

        renderer, mat = _make_renderer(width, height, bg_color, point_size)

        if show_gt and os.path.exists(gt_path):
            gt_pcd = o3d.io.read_point_cloud(gt_path,
                                              remove_nan_points=True,
                                              remove_infinite_points=True)
            gt_pcd.colors = o3d.utility.Vector3dVector(
                np.tile(gt_color, (len(gt_pcd.points), 1))
            )
            renderer.scene.add_geometry("gt", gt_pcd, mat)

        row_panels = []
        for v in views:
            img = _render_frame(renderer, mat, acc_frames[v],
                                 fov, center, radius, elevation, azimuth)
            row_panels.append(img)

        del renderer
        all_panels.append(row_panels)

    # ── Compose figure with matplotlib ─────────────────────────────────────
    px        = 1 / dpi
    fig_w     = width  * n_cols * px + 0.8   # extra left margin for method labels
    fig_h     = height * n_rows * px + 0.4   # extra bottom margin for col labels
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h),
                              gridspec_kw={'hspace': 0.05, 'wspace': 0.02})

    # Normalise axes to 2-D array
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    panel_label_xy = {
        'lower left':  (0.03, 0.03, 'left',  'bottom'),
        'lower right': (0.97, 0.03, 'right', 'bottom'),
        'upper left':  (0.03, 0.97, 'left',  'top'),
        'upper right': (0.97, 0.97, 'right', 'top'),
    }[panel_label_loc]

    for r, (m, row_panels) in enumerate(zip(methods, all_panels)):
        coverages = m.get('coverages', [None] * n_cols)

        for c, (panel, cov) in enumerate(zip(row_panels, coverages)):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])

            # Coverage % above each panel (top-right, like the reference)
            if cov is not None:
                ax.set_title(f'{cov:.1f}%', fontsize=coverage_fontsize,
                              pad=3, loc='right')

            # Per-panel sub-index, e.g. a), b), c) ... in reading order
            if panel_labels:
                idx = r * n_cols + c
                letter = chr(ord('a') + idx) if idx < 26 else f'a{idx - 25}'
                x, y, ha, va = panel_label_xy
                ax.text(x, y, f'{letter})', transform=ax.transAxes,
                        fontsize=panel_label_fontsize, color='black',
                        ha=ha, va=va,
                        bbox=dict(facecolor='white', alpha=0.7,
                                   edgecolor='none', pad=1.5))

            # Dashed separator below every row except the last
            if r < n_rows - 1:
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.plot([0, 1], [0, 0], color='gray', linewidth=0.8,
                        linestyle='--', transform=ax.transAxes, clip_on=False)

        # Method name on the left of the first column
        axes[r, 0].set_ylabel(m['name'], fontsize=method_fontsize,
                               rotation=0, labelpad=6,
                               ha='right', va='center')

    # Column labels at the bottom
    for c, label in enumerate(col_labels):
        axes[-1, c].set_xlabel(label, fontsize=col_label_fontsize, labelpad=4)

    plt.tight_layout(pad=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(output_path + ".png")),
                exist_ok=True)

    png_path = output_path + ".png"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    print(f"[grid] PNG saved → {png_path}")

    eps_path = output_path + ".eps"
    fig.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"[grid] EPS saved → {eps_path}")

    plt.close(fig)
