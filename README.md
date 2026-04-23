# ComfyUI-EquirectProjector

ComfyUI custom nodes for working with 360° equirectangular (ERP) imagery and
video:

1. **Rectilinear → Equirect** — forward gnomonic projection for outpainting a
   normal perspective shot into a full 360° panorama.
2. **Equirect Seam Inpaint Prep / Export** — rolls the horizontal seam into the
   middle of the frame so you can inpaint across it, then rolls it back.

Both workflows are designed to feed diffusion / video models (SDXL, Flux, LTX,
Veo, etc.) that don't natively understand ERP geometry or the ±180° wrap.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Burgstall-labs/ComfyUI-EquirectProjector
cd ComfyUI-EquirectProjector
pip install -r requirements.txt
```

Restart ComfyUI. Nodes appear under the `360/projection` category.

`scipy` is optional — it's used for a fast Euclidean distance transform in the
`bounding_rect` fill path of `Rectilinear → Equirect`. Without it the pack
falls back to a slower pure-PyTorch dilation.

## Nodes

### Rectilinear → Equirect (360 outpaint prep)

Projects a perspective image / video onto a 2:1 equirectangular canvas at a
chosen `(yaw, pitch)`, producing the distorted / padded ERP image plus an
outpaint mask marking the region a diffusion model should generate.

| Input | Description |
|---|---|
| `image` | Source rectilinear `IMAGE` (or IMAGE batch = video) |
| `hfov_deg` | Horizontal field-of-view of the source frame, degrees |
| `equirect_width` / `equirect_height` | Output canvas size (2:1 recommended) |
| `yaw_deg` / `pitch_deg` | Where to place the view on the sphere |
| `shape` | `pincushion` (curved footprint) · `inscribed_rect` (largest axis-aligned rect fully inside) · `bounding_rect` (fill bbox, extrapolating corner gaps from nearest projected pixels) |
| `fill_value` | Scalar `[0,1]` for outside-content pixels |
| `feather_px` | Soft-edge the content/mask boundary |
| `strip_letterbox` / `letterbox_threshold` | Auto-crop black bars before projection |
| `crop_to_239` | Crop input to 2.39:1 before projection (default **on**) |
| `crop_align` | `top` / `center` / `bottom` — vertical anchor for the 2.39 crop |

Outputs `(equirect_image, outpaint_mask)`.

**Defaults target a 2.39:1 @ 100° hfov rectilinear source** (cinemascope spherical).
Feed in any aspect ratio — the node will letterbox-strip and center-crop to 2.39:1
automatically. Disable `crop_to_239` if your model expects a different aspect.

### Equirect Seam Inpaint Prep

Shifts an equirect image by 50% of its width so the wrap-around seam
(columns `0` / `W-1` in the original) lands in the middle of the frame, then
overlays a solid-color stripe of configurable width. The stripe and its mask
are intended as the inpainting region.

| Input | Description |
|---|---|
| `image` | Equirect `IMAGE` |
| `seam_width_px` | Width of the center stripe, pixels |
| `fill_r` / `fill_g` / `fill_b` | Stripe color `[0,1]` |
| `feather_px` | Soft edge on the mask |

Outputs `(shifted_image, shifted_clean_image, seam_mask)`:

- `shifted_image` — the rolled frame **with** the fill stripe painted over the
  seam. Use this as the input to your VAE-encode + inpaint branch.
- `shifted_clean_image` — the rolled frame **without** the stripe (pure
  translation, same pixels as the input just roll-shifted). Use this as the
  clean base in `Equirect Seam Inpaint Compose` or stock `ImageCompositeMasked`
  to avoid the VAE round-trip polluting non-masked pixels.
- `seam_mask` — binary / feathered stripe mask.

> **Breaking change (v0.2):** prep now returns three outputs. Old workflows
> that consumed `(shifted_image, seam_mask)` need to reconnect `seam_mask` to
> the third output socket.

### Equirect Seam Inpaint Export

Rolls the pixels back by `-W/2`, putting the original middle back in the
middle and the seam back at the edges. Pair with *Prep* at the start and
*Export* at the end of the seam-inpainting branch.

Input: `IMAGE`. Output: `IMAGE`.

### Equirect Seam Inpaint Compose (color-match + composite)

Post-decode cleanup node: pastes the inpainted stripe into the clean shifted
base, after optionally colour-matching it to the boundary pixels on each
side of the mask. Fixes two common inference artefacts:

- VAE encode → sample → decode introduces tiny colour drift in unmasked
  regions; this node replaces the unmasked region with the original pixels
  bit-exact.
- The inpaint model sometimes outputs a stripe that is a touch brighter /
  darker / colour-cast than its neighbours. Local strip-sampled colour match
  corrects that without pulling global histograms around.

| Input | Description |
|---|---|
| `inpainted_image` | VAE-decoded post-inpaint frame (still in shifted coords) |
| `clean_shifted_image` | Output 2 of *Prep* — the clean reference |
| `seam_mask` | Output 3 of *Prep* |
| `color_match_mode` | `off` · `mean_shift` · `mean_std` · `boundary_gradient` (default) |
| `match_band_px` | Width of the sampling strip on each side of the mask (default 16) |
| `composite_feather_px` | Extra feather only at composite boundary (default 8) |

Modes:
- `mean_shift` — uniform per-channel offset so the inpaint's boundary mean
  matches the neighbour mean. Safest, never distorts texture.
- `mean_std` — also scales per-channel std (Reinhard-style).
- `boundary_gradient` — measures the left and right boundary offsets
  separately and interpolates a smooth per-column correction across the
  stripe. Best when the two sides of the frame have different colour casts
  (e.g. sun on one side, shade on the other).

Output: `IMAGE`.

### Equirect Seam Latent Prep / Composite / Export

Latent-space counterparts to the pixel-space nodes, for a two-pass workflow
that keeps the first pass' unmasked region bit-identical to the VAE-encoded
original:

- `EquirectSeamLatentPrep` — rolls a `LATENT` by 50% width along the last
  spatial axis, emits a latent-resolution `MASK`. Converts `seam_width_px`
  to latent units via `downsample_factor` (SDXL / SD / LTX image VAE = 8,
  LTX-2 video VAE = 32).
- `EquirectSeamLatentComposite` — `out = base * (1 - mask) + inpainted * mask`
  on latents. Handles both 4D `(B, C, H, W)` and 5D `(B, C, T, H, W)`
  video latents, auto-resizes mask if needed.
- `EquirectSeamLatentExport` — rolls a `LATENT` back by `-W/2`.

## Workflows

### Pixel-space (single-pass, simplest)

```
image ─ Prep ─── shifted_image ────── VAEEncode + Inpaint + VAEDecode ──┐
        ├── shifted_clean_image ─────────────────────────────────────── Compose ─ Export ─ final
        └── seam_mask ──────────────────────────────────────────────────┘
```

The `Compose` step runs the boundary-anchored colour match and composites
into `shifted_clean_image`, so unmasked pixels are preserved bit-exact.

### Latent-space (two-pass, for finicky colour shifts)

```
image ─ VAEEncode ─ L ─ LatentPrep ─┬── L_shifted (clean)                      ┐
                                     │                                          │
                                     ├── KSampler (IC-LoRA, mask) ─ L_inpaint ──┤
                                     │                                          ▼
                                     └── seam_mask_latent ──────── LatentComposite ─ L_clean_shifted
                                                                                     │
                                              LatentExport ─ L_back ◄───────────────┘
                                                  │
                                                  └── (optional) 2nd KSampler,
                                                      standard model, low
                                                      denoise ─ L_refined
                                                          │
                                                          ▼
                                                      VAEDecode ─ final
```

Why two passes: after `LatentComposite`, unmasked latent values are
bit-identical to the encoded original. `LatentExport` then puts the freshly
generated stripe at the image edges (where it belongs). An optional second
KSampler at low denoise on the full frame can smooth any residual
discontinuity where the generated and original latent regions meet.

### Training data for seam-inpaint IC-LoRAs

`Prep → Export` with nothing in between is an identity, so the pair gives
you a cheap way to produce paired "seam-visible" vs "clean" training frames
from any clean equirect source.

## Notes

- All nodes operate on ComfyUI's standard `(B, H, W, C)` IMAGE tensors, `float`
  in `[0, 1]`, and preserve batch dimension (works on video IMAGE batches).
- For the seam nodes, image width doesn't need to be even — the export uses
  the negative-shift inverse so round-trip is exact for any `W`.
- `Rectilinear → Equirect` uses `F.grid_sample` with bilinear filtering and
  runs on the same device as the input tensor.

## License

MIT.
