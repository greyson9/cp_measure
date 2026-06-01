from itertools import product

import numpy
import pytest

from cp_measure.bulk import get_core_measurements, get_core_measurements_3d
from cp_measure.core.measurecolocalization import get_correlation_overlap
from cp_measure.core.measureobjectintensity import (
    INTEGRATED_INTENSITY_EDGE,
    INTENSITY,
    MAX_INTENSITY_EDGE,
    MEAN_INTENSITY_EDGE,
    MIN_INTENSITY_EDGE,
    STD_INTENSITY_EDGE,
    get_intensity,
)
from cp_measure.examples import get_masks, get_pixels


@pytest.mark.parametrize("named_mask", get_masks().items())
@pytest.mark.parametrize("pixels", [get_pixels()])
def test_measurements(named_mask: tuple[str, numpy.ndarray], pixels: numpy.ndarray):
    exceptions = (
        ("one", "feret"),
        ("one", "zernike"),
        *list(  # Downsampling means trouble for the 'thin' masks
            product(
                ("one", "two", "edges", *[f"corner_{i}" for i in range(4)]),
                ("radial_distribution", "radial_zernikes", "texture", "granularity"),
            )
        ),
    )
    mask_name, mask = named_mask
    for name, v in get_core_measurements().items():
        result = v(mask, pixels.copy())
        if (mask_name, name) not in exceptions:
            text = f"Feature {name} returned zero/null on mask {mask_name}"
            if isinstance(result, dict):
                # Test that at least one item contains a valid value
                assert any(
                    [any(~(x == 0 | numpy.isnan(x))) for x in result.values()]
                ), text
                # Test that the output and number of masks match
                assert all([len(x) == mask.max() for x in result.values()]), (
                    f"Input-Output size does not match: Feature {name}, mask {mask_name}"
                )
            else:
                assert result != 0 and not numpy.isnan(result), text


def test_3d_measurements():
    """Test 3D support: 2D-only measurements return empty, 3D ones produce valid output."""
    size = 240
    rng = numpy.random.default_rng(42)
    pixels = rng.integers(low=1, high=255, size=(32, size, size))

    masks = numpy.zeros_like(pixels)
    masks[:, 50:100, 50:100] = 1
    masks[:, 80:120, 90:120] = 1
    masks[:, 150:200, 150:200] = 2
    masks[:, 175:180, 180:210] = 2

    # 2D-only measurements should return empty dict for 3D input
    only_2d = {"radial_distribution", "radial_zernikes", "zernike", "feret"}
    for name, v in get_core_measurements().items():
        result = v(masks, pixels)
        assert isinstance(result, dict), f"{name} did not return a dict"
        if name in only_2d:
            assert result == {}, f"{name} should return empty dict for 3D input"
        else:
            assert any(any(~(x == 0 | numpy.isnan(x))) for x in result.values()), (
                f"{name} returned zero/null on 3D input"
            )
            assert all(len(x) == masks.max() for x in result.values()), (
                f"{name}: output length doesn't match number of objects"
            )

    # get_core_measurements_3d should return only 3D-compatible measurements
    measurements_3d = get_core_measurements_3d()
    assert set(measurements_3d.keys()) == set(get_core_measurements().keys()) - only_2d
    for name, v in measurements_3d.items():
        result = v(masks, pixels)
        assert len(result) > 0, f"{name} returned empty dict"


def test_correlation_overlap():
    size = 240
    rng = numpy.random.default_rng(42)
    pixels = rng.integers(low=1, high=255, size=(size, size, 2))

    # Create two similar-sized objects
    masks = numpy.zeros((size, size), dtype=int)
    masks[50:100, 50:100] = 1  # First square 50x50
    masks[80:120, 90:120] = 1  # Major asymmetries on bottom right edge
    masks[150:200, 150:200] = 2  # Second square 50x50
    masks[175:180, 180:210] = 2  # Minor asymmetries on bottom right edge
    get_correlation_overlap(
        pixels_1=pixels[..., 0], pixels_2=pixels[..., 0], masks=masks
    )


def test_get_intensity_edge_measurements_flag():
    """With edge_measurements=True (default) edge keys are present; with False they are omitted."""
    masks = get_masks()["one"]
    pixels = get_pixels()
    n_objects = int(masks.max())

    edge_keys = [
        f"{INTENSITY}_{INTEGRATED_INTENSITY_EDGE}",
        f"{INTENSITY}_{MEAN_INTENSITY_EDGE}",
        f"{INTENSITY}_{STD_INTENSITY_EDGE}",
        f"{INTENSITY}_{MIN_INTENSITY_EDGE}",
        f"{INTENSITY}_{MAX_INTENSITY_EDGE}",
    ]

    result_with_edge = get_intensity(masks, pixels.copy(), edge_measurements=True)
    for key in edge_keys:
        assert key in result_with_edge, f"edge_measurements=True should include {key}"
        assert len(result_with_edge[key]) == n_objects

    result_without_edge = get_intensity(masks, pixels.copy(), edge_measurements=False)
    for key in edge_keys:
        assert key not in result_without_edge, (
            f"edge_measurements=False should omit {key}"
        )
    assert "Intensity_IntegratedIntensity" in result_without_edge
    assert "Intensity_MeanIntensity" in result_without_edge
    assert all(len(v) == n_objects for v in result_without_edge.values())


def test_intensity_mad_is_dimension_invariant():
    """MADIntensity depends only on the intensity multiset, not on whether the
    object is 2D or 3D.

    Guards a real 3D bug: the MAD quantile rank used ``areas / pixels.ndim``,
    which equals the correct ``areas / 2`` only in 2D and computed a wrong
    (1/3) rank in 3D. Feeding the same 400 values arranged as a 2D vs a 3D
    object must yield the same MAD; pre-fix they differed.
    """
    rng = numpy.random.default_rng(7)
    vals = rng.random(400)

    img2d = vals.reshape(20, 20)
    mask2d = numpy.ones((20, 20), dtype=numpy.int32)
    mad2d = get_intensity(mask2d, img2d)["Intensity_MADIntensity"][0]

    img3d = vals.reshape(4, 10, 10)
    mask3d = numpy.ones((4, 10, 10), dtype=numpy.int32)
    mad3d = get_intensity(mask3d, img3d)["Intensity_MADIntensity"][0]

    assert numpy.isclose(mad2d, mad3d), (
        f"MAD must be dimension-invariant, got 2D={mad2d} vs 3D={mad3d}"
    )


def test_intensity_noncontiguous_labels_raises():
    """Outputs are written at index (label - 1), so non-contiguous labels must
    fail with a clear error rather than an opaque IndexError (README contract)."""
    mask = numpy.zeros((40, 40), dtype=numpy.int32)
    mask[5:15, 5:15] = 1
    mask[20:30, 20:30] = 3  # gap: no label 2
    pixels = get_pixels(size=40)
    with pytest.raises(ValueError, match="contiguous"):
        get_intensity(mask, pixels)


def test_radial_distribution_accepts_bool_mask():
    """A boolean mask is a valid single-object labeling and must be accepted.

    Previously crashed in ``labels.astype(numpy.integer)`` (an abstract type,
    rejected by numpy 2.x)."""
    from cp_measure.core.measureobjectintensitydistribution import (
        get_radial_distribution,
    )

    mask = numpy.zeros((60, 60), dtype=bool)
    mask[10:50, 10:50] = True
    pixels = get_pixels(size=60).astype(float)
    result = get_radial_distribution(mask, pixels)
    assert len(result) > 0
    assert all(len(v) == 1 for v in result.values())


def test_texture_n_jobs_matches_serial():
    """Parallel texture (n_jobs>1) must produce values identical to serial.

    Threading the per-object Haralick calls is purely a performance change; if
    it ever altered an output this test fails.
    """
    from cp_measure.core.measuretexture import get_texture

    rng = numpy.random.default_rng(3)
    mask = numpy.zeros((96, 96), dtype=numpy.int32)
    k = 1
    for r in range(3):
        for c in range(3):
            y, x = 4 + r * 30, 4 + c * 30
            mask[y:y + 20, x:x + 20] = k
            k += 1
    pixels = rng.random((96, 96))

    serial = get_texture(mask, pixels.copy(), n_jobs=1)
    parallel = get_texture(mask, pixels.copy(), n_jobs=4)
    assert serial.keys() == parallel.keys()
    for key in serial:
        numpy.testing.assert_allclose(
            serial[key], parallel[key], equal_nan=True,
            err_msg=f"n_jobs changed output for {key}",
        )
