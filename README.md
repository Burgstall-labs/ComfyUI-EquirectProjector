# ComfyUI-EquirectProjector

ComfyUI custom nodes for working with 360° equirectangular (ERP) imagery and
video. Built primarily for the **LTX2.3 VR-Outpaint** model; may work with
other ERP-aware diffusion / video models too.

1. **Rectilinear → Equirect** — forward gnomonic projection for outpainting a
   normal perspective shot into a full 360° panorama.
2. **Equirect Seam Inpaint Prep / Export** — rolls the horizontal seam into the
   middle of the frame so you can inpaint across it, then rolls it back.
3. **Equirect Seam Inpaint Compose** — post-decode cleanup: boundary-anchored
   local colour-match + feathered composite back onto the clean shifted base.

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
outpaint mask marking the region the model should generate.

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

**Defaults target a 2.39:1 @ 100° hfov rectilinear source** (cinemascope
spherical) — which is what the LTX2.3 VR-Outpaint model was trained on.
Feed in any aspect ratio; the node letterbox-strips and center-crops to 2.39:1
automatically. Disable `crop_to_239` if you're using a different model that
expects another aspect.

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

## Seam-inpainting workflow

```
image ─ Prep ─── shifted_image ────── VAEEncode + Inpaint + VAEDecode ──┐
        ├── shifted_clean_image ─────────────────────────────────────── Compose ─ Export ─ final
        └── seam_mask ──────────────────────────────────────────────────┘
```

`Prep → Export` with nothing in between is an identity — a visible change
only appears when the inpaint branch actually modifies the stripe region.
The pair is also useful for building paired training data for ERP seam-repair
LoRAs / IC-LoRAs: prep gives the "seam in the middle" input, and the clean
equirect source is the target.

## Notes

- All nodes operate on ComfyUI's standard `(B, H, W, C)` IMAGE tensors,
  `float` in `[0, 1]`, and preserve batch dimension (works on video IMAGE
  batches).
- For the seam nodes, image width doesn't need to be even — the export uses
  the negative-shift inverse so round-trip is exact for any `W`.
- `Rectilinear → Equirect` uses `F.grid_sample` with bilinear filtering and
  runs on the same device as the input tensor.

## License

MIT.
