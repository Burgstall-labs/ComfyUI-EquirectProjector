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

Outputs `(shifted_image, seam_mask)`.

### Equirect Seam Inpaint Export

Rolls the pixels back by `-W/2`, putting the original middle back in the
middle and the seam back at the edges. Pair with *Prep* at the start and
*Export* at the end of the seam-inpainting branch.

Input: `IMAGE`. Output: `IMAGE`.

## Seam-inpainting workflow

```
Equirect frame
   │
   ▼
EquirectSeamInpaintPrep ── seam_mask ─┐
   │                                   │
   ▼                                   ▼
  (your inpaint pipeline: VAE encode, KSampler with mask, VAE decode, …)
   │
   ▼
EquirectSeamInpaintExport
   │
   ▼
Final equirect frame (seam continuous, original framing restored)
```

The round-trip `prep → export` with nothing in between is an identity — you'll
only see a visible change when the inpaint pipeline has modified the stripe
region.

The seam-inpaint pair is also useful for building paired training data for
ERP seam-repair LoRAs / IC-LoRAs: prep gives you the "bad" input (visible seam
artifact in center), and a corresponding ground-truth frame can be produced
by running the reverse operation on a clean equirect source.

## Notes

- All nodes operate on ComfyUI's standard `(B, H, W, C)` IMAGE tensors, `float`
  in `[0, 1]`, and preserve batch dimension (works on video IMAGE batches).
- For the seam nodes, image width doesn't need to be even — the export uses
  the negative-shift inverse so round-trip is exact for any `W`.
- `Rectilinear → Equirect` uses `F.grid_sample` with bilinear filtering and
  runs on the same device as the input tensor.

## License

MIT.
