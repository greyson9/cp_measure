# Bug report: get_texture mis-quantizes / crashes on real microscopy images

_From the basicpy_multigpu / img_correction work (2026-06). Intensity, sizeshape,
and granularity measurements are unaffected — this is **texture-only**._

## Location
`src/cp_measure/core/measuretexture.py:250`
```python
pixels = skimage.util.img_as_ubyte(pixels, force_copy=True)
```

## Two problems with `img_as_ubyte` here

1. **Crashes on real-valued float input.** `img_as_ubyte` requires floats in
   `[-1, 1]`; a float image with intensities like `[0, 8435]` raises
   `ValueError: Images of type float must be between -1 and 1`. Callers must
   pre-convert, and that requirement is undocumented.

2. **Silently mis-quantizes uint16 that doesn't fill 0–65535 — i.e. essentially
   all real microscopy.** `img_as_ubyte` scales by the **dtype** range (÷257), so
   a 16-bit image maxing at 8435 collapses to ~33 of 256 gray levels before the
   GLCM, badly degrading the Haralick features with no error raised.

This also **contradicts the docstring** (lines 232–233): *"Before processing, your
image will be rescaled from its current pixel values to 0 – [gray levels − 1]."*
The code rescales by dtype range, not pixel-value range — diverging from both the
docstring and CellProfiler's MeasureTexture (which rescales by the data's min/max).

## Repro
```python
import numpy as np
from cp_measure.bulk import get_core_measurements
M = get_core_measurements()
mask = np.zeros((64, 64), int); mask[10:25, 10:25] = 1
img = (np.random.rand(64, 64) * 8000).astype(np.float32)
M["texture"](mask, img)                      # ValueError: float must be in [-1, 1]
M["texture"](mask, img.astype(np.uint16))    # runs, but quantizes to ~32 levels, not 256
```

## Suggested fix
Rescale by the actual data range to `[0, gray_levels-1]` (matches the docstring +
CellProfiler, accepts any dtype):
```python
m = masks.astype(bool)
lo, hi = pixels[m].min(), pixels[m].max()          # or whole-image min/max
pix = skimage.exposure.rescale_intensity(
    pixels.astype(float), in_range=(lo, hi), out_range=(0, gray_levels - 1)
).astype(np.uint8)
pix[~m] = 0
# the separate `gray_levels != 256` branch can then be dropped
```

## One design decision to make deliberately — normalization *scope*
Whatever range you rescale to, decide whether `in_range` is **per-image** (each
image's own min/max) or a **fixed/global range**. Per-image makes the GLCM gray
levels — and thus texture — non-comparable across images. This is the same *scope*
trap, at a different granularity, that we measured in the image pipeline's 8-bit
rescale: it uses one global range **per plate-run**, and that alone shifted the
*same* object's measured intensity by **−26%** when a brighter well was added to
the plate. For screening, expose `in_range` (or default to a fixed dataset-wide
range) so texture is comparable across the whole dataset, not just within one image.

A regression harness comparing 8-bit vs 16-bit features is available on request.
