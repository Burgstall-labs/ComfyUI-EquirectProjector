import math
import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _max_rect_in_histogram(hist):
    """Largest rectangle in a histogram. Returns (area, left, right_exclusive, height)."""
    stack = []
    best = (0, 0, 0, 0)
    for i, h in enumerate(list(hist) + [0]):
        start = i
        while stack and stack[-1][1] > h:
            s, height = stack.pop()
            area = height * (i - s)
            if area > best[0]:
                best = (area, s, i, height)
            start = s
        stack.append((start, h))
    return best


def _max_inscribed_rect(mask_2d: np.ndarray):
    """Largest axis-aligned rect fully inside a binary mask. Returns (y0, x0, y1, x1) or None."""
    H, W = mask_2d.shape
    heights = np.zeros(W, dtype=np.int32)
    best_area = 0
    best = None
    for y in range(H):
        row = mask_2d[y]
        heights = np.where(row, heights + 1, 0)
        area, x0, x1, h = _max_rect_in_histogram(heights)
        if area > best_area:
            best_area = area
            best = (y - h + 1, x0, y + 1, x1)
    return best


def _bbox(mask_2d: np.ndarray):
    """Axis-aligned bounding box of nonzero region. Returns (y0, x0, y1, x1) or None."""
    ys, xs = np.where(mask_2d)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _crop_to_aspect(image: torch.Tensor, target_aspect: float, align: str = "center") -> torch.Tensor:
    """Crop an (B, H, W, C) IMAGE to `target_aspect = W/H`.

    When the source is taller than the target, removes rows using `align`
    (top/center/bottom). When the source is wider, removes columns using a
    center crop (horizontal alignment is not exposed since it's rarely useful
    for cinema content). No-op if the aspect already matches within 1e-3.
    """
    Hi, Wi = image.shape[1], image.shape[2]
    if Hi == 0 or Wi == 0:
        return image
    current = Wi / Hi
    # Tolerance absorbs common scope-encode rounding (2.387, 2.4, etc.)
    eps = 0.01
    if current < target_aspect - eps:
        new_H = max(1, min(int(round(Wi / target_aspect)), Hi))
        excess = Hi - new_H
        if align == "top":
            y0 = 0
        elif align == "bottom":
            y0 = excess
        else:
            y0 = excess // 2
        return image[:, y0:y0 + new_H, :, :].contiguous()
    if current > target_aspect + eps:
        new_W = max(1, min(int(round(Hi * target_aspect)), Wi))
        x0 = (Wi - new_W) // 2
        return image[:, :, x0:x0 + new_W, :].contiguous()
    return image


def _detect_content_bbox(frame: torch.Tensor, threshold: float):
    """Detect the non-letterbox/pillarbox region of a frame.
    frame: (H, W, 3) float in [0,1]. Returns (top, bottom_excl, left, right_excl)."""
    per_pixel_max = frame.float().max(dim=-1).values
    row_max = per_pixel_max.max(dim=1).values
    col_max = per_pixel_max.max(dim=0).values
    rows = torch.where(row_max > threshold)[0]
    cols = torch.where(col_max > threshold)[0]
    H, W = frame.shape[0], frame.shape[1]
    if len(rows) == 0 or len(cols) == 0:
        return 0, H, 0, W
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _torch_nearest_fill(frames: torch.Tensor, known_mask_2d: torch.Tensor, region_mask_2d: torch.Tensor) -> torch.Tensor:
    """Fill pixels in region_mask but outside known_mask by copying from the nearest
    pixel that IS in known_mask. Works batch-wide with a single EDT pass since the
    nearest-indices depend only on projection geometry, not frame content.

    frames: (B, H, W, 3) float. known_mask_2d, region_mask_2d: (H, W) bool tensors.
    """
    holes = region_mask_2d & (~known_mask_2d)
    if not holes.any():
        return frames
    known_np = known_mask_2d.cpu().numpy()
    if _HAS_SCIPY:
        # Indices of nearest True pixel in known_np for every pixel
        _, (iy, ix) = distance_transform_edt(~known_np, return_indices=True)
        iy_t = torch.from_numpy(iy).to(frames.device).long()
        ix_t = torch.from_numpy(ix).to(frames.device).long()
    else:
        # Pure-torch fallback: iterative dilation with mean of known neighbors.
        # Slower but dependency-free; only runs if scipy is missing.
        return _torch_iterative_fill(frames, known_mask_2d, region_mask_2d)
    # Gather nearest-known values for every pixel, then only write into holes
    H, W = known_mask_2d.shape
    gathered = frames[:, iy_t, ix_t, :]  # (B, H, W, 3)
    holes_e = holes.unsqueeze(0).unsqueeze(-1).to(frames.dtype)  # (1, H, W, 1)
    return frames * (1 - holes_e) + gathered * holes_e


def _torch_iterative_fill(frames: torch.Tensor, known_mask_2d: torch.Tensor, region_mask_2d: torch.Tensor, max_iters: int = 256) -> torch.Tensor:
    """Dilation-based fill: repeatedly set unfilled pixels in the region to the mean of
    their filled 4-neighbors. Only used if scipy is unavailable."""
    device = frames.device
    B, H, W, C = frames.shape
    filled_mask = known_mask_2d.clone().to(torch.float32)
    img = frames.clone()
    target_mask = region_mask_2d.to(torch.float32)
    for _ in range(max_iters):
        if ((target_mask - filled_mask).clamp(min=0).sum() == 0):
            break
        # Pad and compute 4-neighbor mean of filled pixels
        img_p = F.pad(img.permute(0, 3, 1, 2), [1, 1, 1, 1], mode='replicate')
        m_p = F.pad(filled_mask.unsqueeze(0).unsqueeze(0), [1, 1, 1, 1], mode='replicate')
        kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], device=device, dtype=torch.float32)
        kernel_img = kernel.view(1, 1, 3, 3).expand(C, 1, 3, 3)
        neigh_sum_img = F.conv2d(img_p * m_p.expand(B, C, -1, -1), kernel_img, groups=C)
        neigh_sum_m = F.conv2d(m_p, kernel.view(1, 1, 3, 3))
        neigh_mean = neigh_sum_img / neigh_sum_m.clamp(min=1e-6)
        # Newly fillable pixels: inside target, not yet filled, with >= 1 filled neighbor
        newly = (target_mask > 0) & (filled_mask == 0) & (neigh_sum_m[0, 0] > 0)
        if not newly.any():
            break
        newly_e = newly.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1).to(torch.float32)
        img = (img.permute(0, 3, 1, 2) * (1 - newly_e) + neigh_mean * newly_e).permute(0, 2, 3, 1)
        filled_mask = torch.maximum(filled_mask, newly.to(torch.float32))
    return img


class RectilinearToEquirect:
    """Forward gnomonic projection: place a rectilinear view onto an equirect canvas.

    Projects a perspective image/video onto a 2:1 equirectangular panorama at (yaw, pitch),
    producing the distorted/padded equirect image plus an outpaint mask marking the
    region for the diffusion model to generate.

    `shape` options:
      - pincushion: raw forward-projection footprint (curved edges)
      - inscribed_rect: crop content to the largest rect fully inside the pincushion
      - bounding_rect: extend content to the pincushion's bounding rect, extrapolating
                       the corner gaps from the nearest projected pixels
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hfov_deg": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 179.0, "step": 0.5}),
                "equirect_width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 32}),
                "equirect_height": ("INT", {"default": 960, "min": 32, "max": 4096, "step": 32}),
                "yaw_deg": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "pitch_deg": ("FLOAT", {"default": 0.0, "min": -89.0, "max": 89.0, "step": 1.0}),
                "shape": (["pincushion", "inscribed_rect", "bounding_rect"], {"default": "pincushion"}),
                "fill_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather_px": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "strip_letterbox": ("BOOLEAN", {"default": True}),
                "letterbox_threshold": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_to_239": ("BOOLEAN", {"default": True}),
                "crop_align": (["top", "center", "bottom"], {"default": "center"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("equirect_image", "outpaint_mask")
    FUNCTION = "project"
    CATEGORY = "360/projection"

    def project(self, image, hfov_deg, equirect_width, equirect_height,
                yaw_deg, pitch_deg, shape, fill_value, feather_px,
                strip_letterbox=True, letterbox_threshold=0.06,
                crop_to_239=True, crop_align="center"):
        device = image.device
        dtype = image.dtype
        if strip_letterbox and image.shape[0] > 0:
            t, b, l, r = _detect_content_bbox(image[0], float(letterbox_threshold))
            if (t, b, l, r) != (0, image.shape[1], 0, image.shape[2]):
                image = image[:, t:b, l:r, :].contiguous()
        if crop_to_239 and image.shape[0] > 0:
            image = _crop_to_aspect(image, 2.39, crop_align)
        B, Hi, Wi, _ = image.shape
        Weq, Heq = int(equirect_width), int(equirect_height)

        # Equirect lat/lon grid (shape: Heq × Weq)
        lon = (torch.linspace(0, Weq - 1, Weq, device=device, dtype=torch.float32) / Weq - 0.5) * 2 * math.pi
        lat = (0.5 - torch.linspace(0, Heq - 1, Heq, device=device, dtype=torch.float32) / Heq) * math.pi
        lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing='xy')

        lon0 = math.radians(yaw_deg)
        lat0 = math.radians(pitch_deg)
        dlon = lon_grid - lon0

        cos_lat = torch.cos(lat_grid)
        sin_lat = torch.sin(lat_grid)
        cos_dlon = torch.cos(dlon)
        sin_dlon = torch.sin(dlon)
        cos_lat0 = math.cos(lat0)
        sin_lat0 = math.sin(lat0)

        # Forward gnomonic: (lon,lat) → tangent plane (x,y) at (lon0,lat0)
        cos_c = sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_dlon
        visible = cos_c > 1e-6
        cos_c_safe = torch.where(visible, cos_c, torch.ones_like(cos_c))
        x = cos_lat * sin_dlon / cos_c_safe
        y = (cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_dlon) / cos_c_safe

        # Tangent plane → rectilinear pixel coords
        f = (Wi / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        u_rect = x * f + (Wi - 1) / 2.0
        v_rect = -y * f + (Hi - 1) / 2.0

        in_frame = (u_rect >= 0) & (u_rect <= Wi - 1) & (v_rect >= 0) & (v_rect <= Hi - 1)
        pincushion = visible & in_frame  # (Heq, Weq) bool

        # Sample image via grid_sample
        gx = (u_rect / max(Wi - 1, 1)) * 2.0 - 1.0
        gy = (v_rect / max(Hi - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        img_nchw = image.permute(0, 3, 1, 2).contiguous().float()
        sampled = F.grid_sample(img_nchw, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        sampled = sampled.permute(0, 2, 3, 1)  # (B, Heq, Weq, 3)

        # Resolve shape → content mask + image
        pincushion_np = pincushion.cpu().numpy()
        if not pincushion_np.any():
            content_np = np.zeros_like(pincushion_np, dtype=np.float32)
            image_out = sampled
        elif shape == "pincushion":
            content_np = pincushion_np.astype(np.float32)
            image_out = sampled
        elif shape == "inscribed_rect":
            rect = _max_inscribed_rect(pincushion_np)
            content_np = np.zeros_like(pincushion_np, dtype=np.float32)
            if rect is not None:
                y0, x0, y1, x1 = rect
                content_np[y0:y1, x0:x1] = 1.0
            image_out = sampled
        elif shape == "bounding_rect":
            bb = _bbox(pincushion_np)
            content_np = np.zeros_like(pincushion_np, dtype=np.float32)
            if bb is None:
                image_out = sampled
            else:
                y0, x0, y1, x1 = bb
                content_np[y0:y1, x0:x1] = 1.0
                region_mask_t = torch.from_numpy(content_np.astype(bool)).to(device)
                image_out = _torch_nearest_fill(sampled, pincushion, region_mask_t)
        else:
            raise ValueError(f"unknown shape: {shape}")

        content = torch.from_numpy(content_np).to(device=device).unsqueeze(0).unsqueeze(-1)  # (1, Heq, Weq, 1)

        if feather_px > 0:
            k = feather_px * 2 + 1
            m = content.permute(0, 3, 1, 2)
            m = F.pad(m, [feather_px] * 4, mode='replicate')
            m = F.avg_pool2d(m, kernel_size=k, stride=1)
            content = m.permute(0, 2, 3, 1)

        fill = torch.full_like(image_out, float(fill_value))
        out = image_out * content + fill * (1.0 - content)
        out = out.clamp(0.0, 1.0).to(dtype)

        outpaint_mask = (1.0 - content).squeeze(-1).expand(B, -1, -1).contiguous().clamp(0.0, 1.0).to(dtype)
        return (out, outpaint_mask)


class EquirectSeamInpaintPrep:
    """Shift an equirect image by 50% of its width so the wrap-around seam lands
    in the middle, then paint a center stripe the user can inpaint over.

    Pair with EquirectSeamInpaintExport to roll the pixels back after inpainting.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seam_width_px": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 2}),
                "fill_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fill_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fill_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather_px": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("shifted_image", "shifted_clean_image", "seam_mask")
    FUNCTION = "prep"
    CATEGORY = "360/projection"

    def prep(self, image, seam_width_px, fill_r, fill_g, fill_b, feather_px):
        B, H, W, C = image.shape
        device = image.device
        dtype = image.dtype
        shift = W // 2
        shifted = torch.roll(image, shifts=shift, dims=2)

        mask = torch.zeros(H, W, device=device, dtype=torch.float32)
        if seam_width_px > 0:
            half = seam_width_px // 2
            x0 = max(shift - half, 0)
            x1 = min(shift + (seam_width_px - half), W)
            if x1 > x0:
                mask[:, x0:x1] = 1.0

        if feather_px > 0:
            k = feather_px * 2 + 1
            m = mask.unsqueeze(0).unsqueeze(0)
            m = F.pad(m, [feather_px] * 4, mode='replicate')
            m = F.avg_pool2d(m, kernel_size=k, stride=1)
            mask = m.squeeze(0).squeeze(0)

        if C == 3:
            fill = torch.tensor([fill_r, fill_g, fill_b], device=device, dtype=torch.float32).view(1, 1, 1, 3)
        else:
            fill = torch.full((1, 1, 1, C), float(fill_r), device=device, dtype=torch.float32)

        mask_e = mask.unsqueeze(0).unsqueeze(-1)
        painted = shifted.float() * (1.0 - mask_e) + fill * mask_e
        painted = painted.clamp(0.0, 1.0).to(dtype)
        clean = shifted.clamp(0.0, 1.0).to(dtype)

        seam_mask = mask.unsqueeze(0).expand(B, -1, -1).contiguous().clamp(0.0, 1.0).to(dtype)
        return (painted, clean, seam_mask)


class EquirectSeamInpaintExport:
    """Roll equirect pixels by -50% of width, undoing EquirectSeamInpaintPrep so
    the original center column is centered again and the seam returns to the edges."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "export"
    CATEGORY = "360/projection"

    def export(self, image):
        W = image.shape[2]
        return (torch.roll(image, shifts=-(W // 2), dims=2),)


def _mask_xrange(seam_mask_2d: torch.Tensor, threshold: float = 0.5):
    """Return (x_start, x_end_excl) column range of the masked stripe, or None."""
    col_support = seam_mask_2d.float().mean(dim=0)
    hits = torch.where(col_support > threshold)[0]
    if len(hits) == 0:
        return None
    return int(hits.min().item()), int(hits.max().item()) + 1


def _strip_stats(image: torch.Tensor, x0: int, x1: int):
    """Per-channel (mean, std) over the strip [:, :, x0:x1, :]. Returns None if empty."""
    if x1 <= x0:
        return None
    strip = image[:, :, x0:x1, :].float()
    mean = strip.mean(dim=(1, 2))
    std = strip.std(dim=(1, 2), unbiased=False)
    return mean, std


class EquirectSeamInpaintCompose:
    """Composite an inpainted shifted-equirect frame onto the clean shifted
    reference, optionally running a boundary-anchored local color/brightness
    match on the inpainted stripe first.

    Why this exists: VAE round-trip of unmasked regions produces tiny color
    drift, and the IC-LoRA's output doesn't always sit at the same luminance
    as its surroundings. This node fixes both by (1) matching the inpaint's
    stats to a thin strip of neighbor pixels just outside the mask on each
    side, and (2) pasting only the mask region into the unaltered clean base.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inpainted_image": ("IMAGE",),
                "clean_shifted_image": ("IMAGE",),
                "seam_mask": ("MASK",),
                "color_match_mode": (["off", "mean_shift", "mean_std", "boundary_gradient"],
                                     {"default": "boundary_gradient"}),
                "match_band_px": ("INT", {"default": 16, "min": 1, "max": 512, "step": 1}),
                "composite_feather_px": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited_image",)
    FUNCTION = "compose"
    CATEGORY = "360/projection"

    def compose(self, inpainted_image, clean_shifted_image, seam_mask,
                color_match_mode, match_band_px, composite_feather_px):
        dtype = inpainted_image.dtype
        device = inpainted_image.device
        B, H, W, C = inpainted_image.shape

        mask = seam_mask.to(device=device, dtype=torch.float32)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(B, -1, -1)

        xr = _mask_xrange(mask[0])
        corrected = inpainted_image.float()

        if color_match_mode != "off" and xr is not None:
            x_start, x_end = xr
            ln = _strip_stats(clean_shifted_image, max(0, x_start - match_band_px), x_start)
            rn = _strip_stats(clean_shifted_image, x_end, min(W, x_end + match_band_px))
            li = _strip_stats(inpainted_image, x_start, min(x_end, x_start + match_band_px))
            ri = _strip_stats(inpainted_image, max(x_start, x_end - match_band_px), x_end)

            def _shape(v):
                return v.view(B, 1, 1, C)

            if color_match_mode == "mean_shift":
                neigh = [s[0] for s in (ln, rn) if s is not None]
                inp = [s[0] for s in (li, ri) if s is not None]
                if neigh and inp:
                    offset = _shape(torch.stack(neigh).mean(0) - torch.stack(inp).mean(0))
                    corrected[:, :, x_start:x_end, :] = corrected[:, :, x_start:x_end, :] + offset

            elif color_match_mode == "mean_std":
                neigh = [s for s in (ln, rn) if s is not None]
                inp = [s for s in (li, ri) if s is not None]
                if neigh and inp:
                    nm = torch.stack([s[0] for s in neigh]).mean(0)
                    ns = torch.stack([s[1] for s in neigh]).mean(0)
                    im = torch.stack([s[0] for s in inp]).mean(0)
                    isd = torch.stack([s[1] for s in inp]).mean(0)
                    scale = ns / isd.clamp(min=1e-6)
                    stripe = corrected[:, :, x_start:x_end, :]
                    stripe = (stripe - _shape(im)) * _shape(scale) + _shape(nm)
                    corrected[:, :, x_start:x_end, :] = stripe

            elif color_match_mode == "boundary_gradient":
                left_off = None if (ln is None or li is None) else (ln[0] - li[0])
                right_off = None if (rn is None or ri is None) else (rn[0] - ri[0])
                span = x_end - x_start
                if left_off is not None and right_off is not None and span > 1:
                    alpha = torch.linspace(0, 1, span, device=device).view(1, 1, span, 1)
                    offsets = _shape(left_off) + (_shape(right_off) - _shape(left_off)) * alpha
                    corrected[:, :, x_start:x_end, :] = corrected[:, :, x_start:x_end, :] + offsets
                elif left_off is not None or right_off is not None:
                    fallback = left_off if left_off is not None else right_off
                    corrected[:, :, x_start:x_end, :] = corrected[:, :, x_start:x_end, :] + _shape(fallback)

        mask_f = mask
        if composite_feather_px > 0:
            k = composite_feather_px * 2 + 1
            m = mask_f.unsqueeze(1)
            m = F.pad(m, [composite_feather_px] * 4, mode='replicate')
            m = F.avg_pool2d(m, kernel_size=k, stride=1)
            mask_f = m.squeeze(1)

        mask_e = mask_f.unsqueeze(-1).clamp(0.0, 1.0)
        clean = clean_shifted_image.float()
        out = clean * (1.0 - mask_e) + corrected * mask_e
        out = out.clamp(0.0, 1.0).to(dtype)
        return (out,)


class EquirectSeamLatentPrep:
    """Roll a LATENT by 50% of its spatial width so the equirect seam lands in
    the middle, and emit a latent-resolution seam MASK for compositing.

    Operates on the last spatial axis (dim=-1) of `latent["samples"]` — works
    for `(B, C, H, W)` image latents and `(B, C, T, H, W)` video latents.

    Converts `seam_width_px` into latent units via `downsample_factor` (VAE
    spatial compression: SDXL/SD=8, LTX image VAE=8, LTX-2 video VAE=32).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "seam_width_px": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 2}),
                "downsample_factor": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "feather_latent_px": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("shifted_latent", "seam_mask_latent")
    FUNCTION = "prep"
    CATEGORY = "360/projection"

    def prep(self, latent, seam_width_px, downsample_factor, feather_latent_px):
        samples = latent["samples"]
        if samples.ndim < 4:
            raise ValueError(f"expected LATENT with rank >= 4 (B,C,...,H,W), got {samples.shape}")
        W_lat = samples.shape[-1]
        H_lat = samples.shape[-2]
        B = samples.shape[0]
        shift = W_lat // 2
        shifted = torch.roll(samples, shifts=shift, dims=-1)

        seam_w_lat = max(0, int(round(seam_width_px / max(1, downsample_factor))))
        mask = torch.zeros(H_lat, W_lat, dtype=torch.float32, device=samples.device)
        if seam_w_lat > 0:
            half = seam_w_lat // 2
            x0 = max(shift - half, 0)
            x1 = min(shift + (seam_w_lat - half), W_lat)
            if x1 > x0:
                mask[:, x0:x1] = 1.0

        if feather_latent_px > 0:
            k = feather_latent_px * 2 + 1
            m = mask.unsqueeze(0).unsqueeze(0)
            m = F.pad(m, [feather_latent_px] * 4, mode='replicate')
            m = F.avg_pool2d(m, kernel_size=k, stride=1)
            mask = m.squeeze(0).squeeze(0)

        seam_mask_latent = mask.unsqueeze(0).expand(B, -1, -1).contiguous().clamp(0.0, 1.0)
        out_latent = {**latent, "samples": shifted}
        return (out_latent, seam_mask_latent)


class EquirectSeamLatentComposite:
    """Composite two latents with a mask: `out = base * (1-m) + inpainted * m`.

    Use post first-pass KSampler to keep unmasked regions pixel-identical to
    the encoded original (kills VAE-round-trip drift in the non-seam region).

    Accepts MASK at latent resolution (from `EquirectSeamLatentPrep`); if a
    different size is passed, it is bilinearly resized to the samples' spatial
    shape.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_latent": ("LATENT",),
                "inpainted_latent": ("LATENT",),
                "seam_mask_latent": ("MASK",),
                "time_mode": (["slice_inp_last", "slice_inp_first",
                               "slice_inp_center", "pad_base", "strict"],
                              {"default": "slice_inp_last"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("composited_latent",)
    FUNCTION = "composite"
    CATEGORY = "360/projection"

    def composite(self, base_latent, inpainted_latent, seam_mask_latent, time_mode="slice_inp_last"):
        base = base_latent["samples"]
        inp = inpainted_latent["samples"]

        # Reconcile a time-dim mismatch on 5D video latents. LTX samplers often
        # output more frames than the encoded input (e.g. 13 → 26 for
        # conditioning + denoised concatenation). The leading conditioning
        # frames are in original coords, the trailing denoised frames are in
        # shifted coords — so for a correct Export roll-back, keep only the
        # shifted section.
        #
        # slice_inp_last (default): keep the last base_T frames of inp — the
        #   denoised section, which IS in shifted coords and rolls back
        #   correctly via LatentExport.
        # slice_inp_first/center: same idea, different window.
        # pad_base: keep inp's full T. Leading frames pass through as-is.
        #   WARNING: for LTX-style samplers the leading frames are conditioning
        #   in original coords — Export will incorrectly roll them and you'll
        #   see shifted content in that half of the decoded video. Only use
        #   this if you know the sampler output is uniformly in shifted coords.
        # strict: raise on any mismatch.
        if (base.ndim == 5 and inp.ndim == 5
                and base.shape[:2] == inp.shape[:2]
                and base.shape[-2:] == inp.shape[-2:]
                and base.shape[2] != inp.shape[2]):
            base_T, inp_T = base.shape[2], inp.shape[2]
            if base_T == 0:
                raise ValueError(
                    f"base_latent has T=0 but inpainted_latent has T={inp_T}. "
                    f"Upstream base is empty — check that base_latent is "
                    f"connected to EquirectSeamLatentPrep's shifted_latent "
                    f"output (not a KSampler output)."
                )
            if time_mode == "pad_base" and inp_T > base_T:
                # Use inp's leading (inp_T - base_T) frames as the "pad" so
                # the composite formula out = base*(1-m)+inp*m yields inp for
                # those frames (no-op compositing there).
                leading = inp[:, :, :inp_T - base_T, :, :]
                base = torch.cat([leading, base], dim=2)
            elif time_mode == "slice_inp_last" and inp_T >= base_T:
                inp = inp[:, :, inp_T - base_T:, :, :]
            elif time_mode == "slice_inp_first" and inp_T >= base_T:
                inp = inp[:, :, :base_T, :, :]
            elif time_mode == "slice_inp_center" and inp_T >= base_T:
                start = (inp_T - base_T) // 2
                inp = inp[:, :, start:start + base_T, :, :]
            # "strict" or unhandled direction (e.g. base_T > inp_T in
            # pad_base) falls through to the error below.

        if base.shape != inp.shape:
            raise ValueError(
                f"latent shape mismatch: base {tuple(base.shape)} vs inpainted "
                f"{tuple(inp.shape)}. If only the T dim differs, pick a "
                f"time_mode that handles your direction (currently {time_mode!r})."
            )
        if any(d == 0 for d in base.shape):
            raise ValueError(
                f"composited latent would have a zero-sized dim {tuple(base.shape)}. "
                f"Inspect upstream: base_latent or inpainted_latent is empty."
            )

        m = seam_mask_latent.to(device=base.device, dtype=base.dtype)
        if m.ndim == 2:
            m = m.unsqueeze(0)

        target_H, target_W = base.shape[-2], base.shape[-1]
        if m.shape[-2] != target_H or m.shape[-1] != target_W:
            m4 = m.view(m.shape[0], 1, m.shape[-2], m.shape[-1])
            m4 = F.interpolate(m4, size=(target_H, target_W), mode='bilinear', align_corners=False)
            m = m4.squeeze(1)

        # Broadcast: base is (B, C, H, W) or (B, C, T, H, W). Insert singleton
        # dims between batch and spatial axes so m lines up with any rank.
        extra = base.ndim - 3  # axes between batch and last-2 spatial dims
        for _ in range(extra):
            m = m.unsqueeze(1)

        m = m.clamp(0.0, 1.0)
        out = base * (1.0 - m) + inp * m
        return ({**base_latent, "samples": out},)


class EquirectSeamLatentExport:
    """Roll a LATENT by -50% of its spatial width, undoing EquirectSeamLatentPrep."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "export"
    CATEGORY = "360/projection"

    def export(self, latent):
        samples = latent["samples"]
        W_lat = samples.shape[-1]
        shifted = torch.roll(samples, shifts=-(W_lat // 2), dims=-1)
        return ({**latent, "samples": shifted},)


NODE_CLASS_MAPPINGS = {
    "RectilinearToEquirect": RectilinearToEquirect,
    "EquirectSeamInpaintPrep": EquirectSeamInpaintPrep,
    "EquirectSeamInpaintExport": EquirectSeamInpaintExport,
    "EquirectSeamInpaintCompose": EquirectSeamInpaintCompose,
    "EquirectSeamLatentPrep": EquirectSeamLatentPrep,
    "EquirectSeamLatentComposite": EquirectSeamLatentComposite,
    "EquirectSeamLatentExport": EquirectSeamLatentExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RectilinearToEquirect": "Rectilinear → Equirect (360 outpaint prep)",
    "EquirectSeamInpaintPrep": "Equirect Seam Inpaint Prep",
    "EquirectSeamInpaintExport": "Equirect Seam Inpaint Export",
    "EquirectSeamInpaintCompose": "Equirect Seam Inpaint Compose (color-match + composite)",
    "EquirectSeamLatentPrep": "Equirect Seam Latent Prep",
    "EquirectSeamLatentComposite": "Equirect Seam Latent Composite",
    "EquirectSeamLatentExport": "Equirect Seam Latent Export",
}
