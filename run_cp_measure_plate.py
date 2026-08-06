#!/usr/bin/env python
"""run_cp_measure_plate.py — run cp_measure over a plate of fields, in parallel.

Replaces the *measurement* step of a CellProfiler analysis pipeline
(MeasureObjectIntensity / MeasureObjectSizeShape[+Zernike+Feret] /
MeasureTexture / MeasureGranularity / MeasureObjectIntensityDistribution) for
the Cells and Nuclei objects defined by cellpose label masks. It also derives
Cytoplasm (IdentifyTertiary) and emits CellProfiler's relational columns
(Cytoplasm.Parent_Cells/Parent_Nuclei, Children_Cytoplasm_Count) and per-cell
LoG spot counts (Children_Spots*_Count). It does NOT do segmentation or object
filtering — run those separately.

Fields are discovered by globbing the cell masks; the raw channels are paired
by the ``_w<N>`` token in their filenames. Intensity images are scaled to
[0, 1] by their integer dtype max (uint8 -> /255, uint16 -> /65535) to match
CellProfiler's "Set intensity range from image metadata". Labels are made
contiguous with ``relabel_sequential`` (cp_measure requires 1..N).

Each field runs in its own process; texture uses n_jobs=1 so it does not
oversubscribe the per-field parallelism. One table is written per
(field, object set): ``<out>/<well>_s<site>_<ObjectSet>.<parquet|csv>``,
one row per object, columns named CellProfiler-style.

Example
-------
    python run_cp_measure_plate.py \\
        --images-dir /.../image_correction/rescaled_imgs \\
        --masks-dir  /.../20260421_compound_plate46 \\
        --output-dir /.../cp_measure_out --workers 16

Edit the CONFIG block for a new plate's channel tokens / mask names.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tifffile
from scipy.ndimage import gaussian_laplace, grey_dilation, grey_erosion
from skimage.feature import blob_log
from skimage.segmentation import expand_labels, relabel_sequential

from cp_measure.core.measuregranularity import get_granularity
from cp_measure.core.measureobjectintensity import get_intensity
from cp_measure.core.measureobjectintensitydistribution import get_radial_distribution
from cp_measure.multimask.measureobjectneighbors import measureobjectneighbors
from cp_measure.core.measureobjectsizeshape import get_feret, get_sizeshape, get_zernike
from cp_measure.core.measuretexture import get_texture

# Quiet two known-benign, high-volume warnings so genuine per-field errors (counted
# and printed separately) stay visible: (1) cp_measure's radial distribution does
# 0/0 divides for objects with no pixels in a bin / no valid center, yielding the
# intended NaN; (2) scikit-image renamed RegionProperties['intensity_image'] ->
# 'image_intensity' (cosmetic until skimage 2.0). Inherited by worker processes.
np.seterr(divide="ignore", invalid="ignore")
warnings.filterwarnings("ignore", message=r".*intensity_image.*", category=FutureWarning)

# ----------------------------- CONFIG (edit per plate) -----------------------
CHANNEL_TOKENS = {"Cy5": "w1", "Cy3": "w2", "FITC": "w3", "DAPI": "w4"}  # channel -> filename token
OBJECT_MASKS = {"Cells": "cellmasks", "Nuclei": "nucleimasks"}          # object set -> mask name
FIELD_RE = re.compile(r"_([A-Za-z]\d+)_s(\d+)_")                        # -> (well, site)

TEXTURE_SCALES = [3, 5, 10]
TEXTURE_GRAY = 32
GRANULARITY_LEN = 16
RADIAL_BINS = 4
# Radial-distribution center: object set -> object set whose centroids center
# the radial bins (CellProfiler "Centers of other objects"). E.g. each Cell's
# radial is centered on its Nucleus. Sets not listed (or whose center set is
# not loaded) self-center. The center set must be in --objects to be loaded.
RADIAL_CENTER = {"Cells": "Nuclei", "Cytoplasm": "Nuclei"}
# Objects derived from others (CellProfiler IdentifyTertiaryObjects): name ->
# (larger, smaller). Cytoplasm = Cells minus the interior of Nuclei (keeping
# the nucleus outline), matching CP's "shrink smaller object". Both sets must
# be in --objects so their masks are loaded.
DERIVED_OBJECTS = {"Cytoplasm": ("Cells", "Nuclei")}
# Neighbors (CellProfiler MeasureObjectNeighbors): object set -> list of
# (method, distance). Each runs self-neighbors; cp_measure already names the
# columns Neighbors_<feature>_<Adjacent|distance> to match CellProfiler. Border
# objects may differ slightly from CP (which discards border-touching objects).
NEIGHBORS = {"Cells": [("Adjacent", 5), ("Within a specified distance", 100)]}
# Relational aggregation ("relate"). The only non-spot RelateObjects that reaches
# CellProfiler's export is the IdentifyTertiary (Cytoplasm) relationship:
# Cytoplasm gets Parent_Cells/Parent_Nuclei; Cells & Nuclei get
# Children_Cytoplasm_Count. (CP's CellsCP<-NucleiCP per-parent means ran on the
# pre-filter objects and are absent from Cells.csv, so we do NOT emit Mean_Nuclei_*.)
# Needs Cytoplasm in --objects.
# Spot detection ("spots"): LoG with a per-channel threshold calibrated to each
# channel's matched-noise floor (validated to track CP's dose-dependent organelle
# spot biology: CCCP/rotenone/quinidine etc.). SPOT_K is the noise-sigma multiplier.
# Each detected centre is grown into a disk (SPOT_RADIUS) and measured with
# cp_measure; per cell we emit Children_Spots*_Count plus the Mean of the full
# intensity panel and two interpretable morphology features (area, eccentricity).
# Spot shape beyond those added no dose signal in validation, so it is excluded.
SPOT_SIGMA = {"Cy3": (2.0, 5.5, 3), "Cy5": (2.0, 5.5, 3), "FITC": (1.0, 4.0, 4), "DAPI": (1.5, 5.0, 3)}
SPOT_K = {"Cy3": 5.0, "Cy5": 6.0, "DAPI": 8.0, "FITC": 15.0}
SPOT_RADIUS = 4
SPOT_SHAPE_FEATURES = ["AreaShape_Area", "AreaShape_Eccentricity"]
SPOT_TYPES = {"SpotsCy3Cyto": ("Cy3", "cyto"), "SpotsCy3Nuc": ("Cy3", "nuc"),
              "SpotsCy5Cyto": ("Cy5", "cyto"), "SpotsCy5Nuc": ("Cy5", "nuc"),
              "SpotsFITCCyto": ("FITC", "cyto"), "SpotsFITCNuc": ("FITC", "nuc"),
              "SpotsNucDAPI": ("DAPI", "nuc")}
ALL_FEATURES = ["shape", "intensity", "texture", "granularity", "radial", "neighbors", "relate", "spots"]

DEFAULT_IMAGES = "/home/grlewis/Projects/1017/output/20260421_compound_plate46/image_correction/rescaled_imgs"
DEFAULT_MASKS = "/home/grlewis/Projects/1017/output/20260421_compound_plate46/segmentation/cellpose"
# -----------------------------------------------------------------------------


def _normalize(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.integer):
        return img.astype(np.float32) / float(np.iinfo(img.dtype).max)
    return img.astype(np.float32)


def _derive_tertiary(larger, smaller):
    """CellProfiler IdentifyTertiaryObjects (shrink smaller object): larger minus
    the interior of smaller, keeping smaller's 1px outline; retains larger's
    labels. Matches CP to the pixel except at a few image-border objects."""
    fp = np.ones((3, 3))
    outline = (smaller > 0) & (
        (smaller != grey_erosion(smaller, footprint=fp))
        | (smaller != grey_dilation(smaller, footprint=fp))
    )
    tertiary = larger.copy()
    tertiary[(smaller > 0) & ~outline] = 0
    return tertiary


def max_overlap_map(a, b):
    """For each label in `a` (1..max), the label in `b` with maximal pixel overlap
    (0 if it overlaps no nonzero `b` label). CellProfiler's parent-by-overlap rule."""
    na = int(a.max())
    out = np.zeros(na + 1, dtype=np.int64)
    if na == 0 or b is None:
        return out
    nb = int(b.max())
    fa = a.ravel()
    sel = fa > 0
    ka = fa[sel].astype(np.int64)
    kb = b.ravel()[sel].astype(np.int64)
    key = ka * (nb + 1) + kb
    uk, cnt = np.unique(key, return_counts=True)
    best = np.zeros(na + 1, dtype=np.int64)
    for k, n in zip(uk.tolist(), cnt.tolist()):
        ai, bi = divmod(k, nb + 1)
        if bi != 0 and n > best[ai]:
            best[ai] = n
            out[ai] = bi
    return out


def _append(result, items):
    """Append (column_name, 1d-array) pairs to a featurized object-set result."""
    if not items:
        return
    result["cols"] += [c for c, _ in items]
    result["data"] = np.column_stack([result["data"]] + [np.asarray(a, float) for _, a in items])


def add_relate_columns(results, masks, cyto_to_cell):
    """CellProfiler IdentifyTertiary relationships (the only non-spot ones in CP's
    export): Cytoplasm.Parent_Cells/Parent_Nuclei + Children_Cytoplasm_Count on
    Cells & Nuclei."""
    if "Cytoplasm" not in masks or "Cells" not in masks:
        return
    cells, nuclei, cyto = masks["Cells"], masks.get("Nuclei"), masks["Cytoplasm"]
    inv = cyto_to_cell.get("Cytoplasm")  # new cyto label -> cell label
    cell_to_nuc = max_overlap_map(cells, nuclei) if nuclei is not None else None
    cyto_cell = inv[np.asarray(results["Cytoplasm"]["ids"], int)] if "Cytoplasm" in results else np.array([], int)

    if "Cytoplasm" in results:
        items = [("Parent_Cells", cyto_cell.astype(float))]
        if cell_to_nuc is not None:
            items.append(("Parent_Nuclei", cell_to_nuc[cyto_cell].astype(float)))
        _append(results["Cytoplasm"], items)
    if "Cells" in results:
        has_cyto = set(int(c) for c in cyto_cell)
        cnt = np.array([1.0 if i in has_cyto else 0.0 for i in results["Cells"]["ids"]])
        _append(results["Cells"], [("Children_Cytoplasm_Count", cnt)])
    if nuclei is not None and "Nuclei" in results:
        nuc_to_cell = max_overlap_map(nuclei, cells)   # nucleus -> containing cell (enables joins to Cells)
        parent_cells = np.array([float(nuc_to_cell[j]) for j in results["Nuclei"]["ids"]])
        nuc_of_cyto = cell_to_nuc[cyto_cell]
        cc = np.bincount(nuc_of_cyto[nuc_of_cyto > 0], minlength=int(nuclei.max()) + 1)
        cnt = np.array([float(cc[j]) for j in results["Nuclei"]["ids"]])
        _append(results["Nuclei"], [("Parent_Cells", parent_cells), ("Children_Cytoplasm_Count", cnt)])


def _spot_feature_schema():
    """Per-spot features averaged per parent: the full intensity panel (minus
    positional X/Y entries) plus the curated shape features."""
    m = np.zeros((24, 24), np.int32)
    m[6:18, 6:18] = 1
    inten = get_intensity(m, np.random.default_rng(0).random((24, 24)).astype(np.float32))
    keys = [k for k in inten if not (k.endswith("_X") or k.endswith("_Y"))]
    return keys + SPOT_SHAPE_FEATURES


SPOT_FEATURES = _spot_feature_schema()


def _agg_mean(parent, vals, n):
    """Per-parent mean of `vals`; NaN where a parent has no finite child spot."""
    keep = (parent > 0) & np.isfinite(vals)
    s = np.bincount(parent[keep] - 1, weights=vals[keep], minlength=n)
    c = np.bincount(parent[keep] - 1, minlength=n)
    out = np.full(n, np.nan)
    nz = c > 0
    out[nz] = s[nz] / c[nz]
    return out


def measure_spots(channels, cells, nuclei):
    """Detect LoG spots, grow each centre to a disk (SPOT_RADIUS), measure with
    cp_measure, and aggregate per parent. Returns {"Cells"|"Nuclei"|"Cytoplasm":
    {column: array}} with Children_<st>_Count and Mean_<st>_<feature> over the full
    SPOT_FEATURES set (Means NaN where a parent has no such spot). Cells/Cytoplasm
    arrays are indexed by cell label, Nuclei by nucleus label."""
    nc, nn = int(cells.max()), int(nuclei.max())
    incell = cells > 0
    sizes = {"Cells": nc, "Nuclei": nn, "Cytoplasm": nc}
    parent_types = {
        "Cells": list(SPOT_TYPES),
        "Nuclei": [s for s, (c, comp) in SPOT_TYPES.items() if comp == "nuc"],
        "Cytoplasm": [s for s, (c, comp) in SPOT_TYPES.items() if comp == "cyto"],
    }
    out = {p: {} for p in sizes}
    for p, sts in parent_types.items():
        for st in sts:
            out[p][f"Children_{st}_Count"] = np.zeros(sizes[p])
            for f in SPOT_FEATURES:
                out[p][f"Mean_{st}_{f}"] = np.full(sizes[p], np.nan)
    det = {}
    for ch in dict.fromkeys(c for c, _ in SPOT_TYPES.values()):
        if ch not in channels:
            continue
        mn, mx, ns = SPOT_SIGMA[ch]
        resp = -gaussian_laplace(channels[ch], mn) * (mn ** 2)
        r = resp[incell]
        sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
        b = blob_log(channels[ch], min_sigma=mn, max_sigma=mx, num_sigma=ns, threshold=SPOT_K[ch] * sigma, threshold_rel=None)
        if len(b):
            y, x = b[:, 0].astype(int), b[:, 1].astype(int)
            inc = cells[y, x] > 0
            y, x = y[inc], x[inc]
            det[ch] = (y, x, nuclei[y, x] > 0)
        else:
            det[ch] = (np.array([], int), np.array([], int), np.array([], bool))
    for st, (ch, comp) in SPOT_TYPES.items():
        if ch not in det:
            continue
        y, x, in_nuc = det[ch]
        sel = in_nuc if comp == "nuc" else ~in_nuc
        ys, xs = y[sel], x[sel]
        if not len(ys):
            continue
        out["Cells"][f"Children_{st}_Count"] = np.bincount(cells[ys, xs] - 1, minlength=nc).astype(float)
        centers = np.zeros(cells.shape, np.int32)
        centers[ys, xs] = np.arange(1, len(ys) + 1)
        smask = np.where(incell, expand_labels(centers, distance=SPOT_RADIUS), 0)
        smask = relabel_sequential(smask)[0].astype(np.int32)
        if smask.max() == 0:
            continue
        inten = get_intensity(smask, channels[ch].copy())
        ss = get_sizeshape(smask, None, calculate_advanced=False)
        feats = {k: np.asarray(v, float) for k, v in inten.items() if not (k.endswith("_X") or k.endswith("_Y"))}
        feats["AreaShape_Area"] = np.asarray(ss["Area"], float)
        feats["AreaShape_Eccentricity"] = np.asarray(ss["Eccentricity"], float)
        cyc = np.clip(np.round(np.asarray(ss["Center_Y"])).astype(int), 0, cells.shape[0] - 1)
        cxc = np.clip(np.round(np.asarray(ss["Center_X"])).astype(int), 0, cells.shape[1] - 1)
        pcell = cells[cyc, cxc]
        for f, v in feats.items():
            out["Cells"][f"Mean_{st}_{f}"] = _agg_mean(pcell, v, nc)
        if comp == "nuc":
            pnuc = nuclei[cyc, cxc]
            out["Nuclei"][f"Children_{st}_Count"] = np.bincount(pnuc[pnuc > 0] - 1, minlength=nn).astype(float)
            for f, v in feats.items():
                out["Nuclei"][f"Mean_{st}_{f}"] = _agg_mean(pnuc, v, nn)
        else:
            out["Cytoplasm"][f"Children_{st}_Count"] = np.bincount(pcell[pcell > 0] - 1, minlength=nc).astype(float)
            for f, v in feats.items():
                out["Cytoplasm"][f"Mean_{st}_{f}"] = _agg_mean(pcell, v, nc)
    return out


def add_spot_measurements(results, spots, cyto_inv):
    """Attach Children_<st>_Count + Mean_<st>_<feature> columns to each parent."""
    for obj_set in ("Cells", "Nuclei", "Cytoplasm"):
        if obj_set not in results or obj_set not in spots:
            continue
        ids = np.asarray(results[obj_set]["ids"], int)
        if obj_set == "Cytoplasm":
            if cyto_inv is None:
                continue
            idx = cyto_inv[ids] - 1            # cytoplasm -> its cell label
        else:
            idx = ids - 1
        items = []
        for col in sorted(spots[obj_set]):
            arr = spots[obj_set][col]
            valid = (idx >= 0) & (idx < len(arr))
            fill = 0.0 if col.startswith("Children_") else np.nan
            vals = np.where(valid, arr[np.clip(idx, 0, max(len(arr) - 1, 0))], fill)
            items.append((col, vals))
        _append(results[obj_set], items)



def featurize_object_set(obj_set, mask, channels, features, in_range, radial_center=None):
    """Return (columns, data[obj, feat], object_ids, center_x, center_y)."""
    ss = get_sizeshape(mask, None)
    cx, cy = np.asarray(ss["Center_X"], float), np.asarray(ss["Center_Y"], float)
    cols, data = [], []

    if "shape" in features:
        for src in (ss, get_zernike(mask, None), get_feret(mask, None)):
            for k, v in src.items():
                cols.append(f"AreaShape_{k}")
                data.append(np.asarray(v, float))

    if "neighbors" in features and obj_set in NEIGHBORS:
        for method, dist in NEIGHBORS[obj_set]:
            # cp_measure already names keys Neighbors_<feature>_<Adjacent|distance>,
            # matching CellProfiler's column suffixes.
            for k, v in measureobjectneighbors(mask, mask, distance_method=method, distance=dist).items():
                cols.append(k)
                data.append(np.asarray(v, float))

    for ch, img in channels.items():
        if "intensity" in features:
            for k, v in get_intensity(mask, img.copy()).items():
                cols.append(f"{k}_{ch}")
                data.append(np.asarray(v, float))
        if "texture" in features:
            for sc in TEXTURE_SCALES:
                for k, v in get_texture(mask, img, scale=sc, in_range=in_range, 
                                        gray_levels=TEXTURE_GRAY, n_jobs=1).items():
                    feat, rest = k.split("_", 1)
                    cols.append(f"Texture_{feat}_{ch}_{rest}")
                    data.append(np.asarray(v, float))
        if "granularity" in features:
            for k, v in get_granularity(mask, img, granular_spectrum_length=GRANULARITY_LEN).items():
                cols.append(f"{k}_{ch}")
                data.append(np.asarray(v, float))
        if "radial" in features:
            for k, v in get_radial_distribution(
                mask, img, bin_count=RADIAL_BINS, center_labels=radial_center
            ).items():
                meas, b = k.replace("RadialDistribution_", "").rsplit("_", 1)
                cols.append(f"RadialDistribution_{meas}_{ch}_{b}")
                data.append(np.asarray(v, float))

    block = np.column_stack(data) if data else np.empty((len(cx), 0))
    return cols, block, np.arange(1, len(cx) + 1), cx, cy


def write_table(path, well, site, obj_set, cols, data, ids, cx, cy, fmt):
    n = len(ids)
    meta = {
        "Metadata_Well": [well] * n,
        "Metadata_Site": [site] * n,
        "ObjectSet": [obj_set] * n,
        "ObjectNumber": ids.tolist(),
        "Center_X": cx.tolist(),
        "Center_Y": cy.tolist(),
    }
    if fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = {**meta, **{c: data[:, i] for i, c in enumerate(cols)}}
        pq.write_table(pa.table(table), path)
    else:
        import csv

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(meta) + cols)
            for r in range(n):
                w.writerow([meta[k][r] for k in meta] + data[r].tolist())


def process_field(task):
    field, outdir, features, fmt, objects, in_range = task
    t = time.perf_counter()
    try:
        channels = {ch: _normalize(tifffile.imread(p)) for ch, p in field["channels"].items()}
        masks = {}
        for obj_set, mpath in field["masks"].items():
            m = relabel_sequential(tifffile.imread(mpath))[0].astype(np.int32)
            if m.max() > 0:
                masks[obj_set] = m
        # derive tertiary objects (e.g. Cytoplasm = Cells - Nuclei interior). Relabel
        # to contiguous (cp_measure needs 1..N) and keep new-label -> cell-label map.
        cyto_to_cell = {}
        for obj_set in objects:
            if obj_set in DERIVED_OBJECTS and obj_set not in masks:
                larger, smaller = DERIVED_OBJECTS[obj_set]
                if larger in masks and smaller in masks:
                    relab, _, inv = relabel_sequential(_derive_tertiary(masks[larger], masks[smaller]))
                    if relab.max() > 0:
                        masks[obj_set] = relab.astype(np.int32)
                        cyto_to_cell[obj_set] = np.asarray(inv)
        # featurize every object set, holding results so cross-object columns
        # (relate, spot counts) can be attached before writing.
        results = {}
        for obj_set in objects:
            if obj_set not in masks:
                continue
            center_set = RADIAL_CENTER.get(obj_set)
            radial_center = masks.get(center_set) if center_set else None
            cols, data, ids, cx, cy = featurize_object_set(obj_set, masks[obj_set], channels, 
                                                           features, in_range, radial_center)
            results[obj_set] = {"cols": list(cols), "data": data, "ids": ids, "cx": cx, "cy": cy}
        if "relate" in features:
            add_relate_columns(results, masks, cyto_to_cell)
        if "spots" in features and "Cells" in masks and "Nuclei" in masks:
            add_spot_measurements(results, measure_spots(channels, masks["Cells"], masks["Nuclei"]),
                                  cyto_to_cell.get("Cytoplasm"))
        n_obj = 0
        ext = "parquet" if fmt == "parquet" else "csv"
        for obj_set in objects:
            if obj_set not in results:
                continue
            r = results[obj_set]
            out = os.path.join(outdir, f"{field['id']}_{obj_set}.{ext}")
            write_table(out, field["well"], field["site"], obj_set, r["cols"], r["data"], r["ids"], r["cx"], r["cy"], fmt)
            n_obj += len(r["ids"])
        return field["id"], "ok", n_obj, time.perf_counter() - t, ""
    except Exception as e:  # noqa: BLE001 - report per field, keep the plate going
        return field["id"], "error", 0, time.perf_counter() - t, f"{type(e).__name__}: {e}"


def discover_fields(images_dir, masks_dir, objects, limit):
    primary = OBJECT_MASKS[objects[0]]
    fields = {}
    for mp in glob.glob(os.path.join(masks_dir, "**", f"*{primary}.tif"), recursive=True):
        m = FIELD_RE.search(os.path.basename(mp))
        if not m:
            continue
        well, site = m.group(1), int(m.group(2))
        fid = f"{well}_s{site}"
        if fid in fields:
            continue
        chans, ok = {}, True
        for ch, tok in CHANNEL_TOKENS.items():
            g = glob.glob(os.path.join(images_dir, "**", f"*_{well}_s{site}_{tok}*.tif"), recursive=True)
            if not g:
                ok = False
                break
            chans[ch] = g[0]
        if not ok:
            continue
        masks = {}
        for obj in objects:
            g = glob.glob(os.path.join(masks_dir, "**", f"*_{well}_s{site}_*{OBJECT_MASKS[obj]}.tif"), recursive=True)
            if g:
                masks[obj] = g[0]
        if len(masks) == len(objects):
            fields[fid] = {"id": fid, "well": well, "site": site, "channels": chans, "masks": masks}
    out = sorted(fields.values(), key=lambda f: f["id"])
    return out[:limit] if limit else out


def resolve_format(fmt):
    if fmt != "auto":
        return fmt
    try:
        import pyarrow  # noqa: F401

        return "parquet"
    except ImportError:
        return "csv"


def _ext(fmt):
    return "parquet" if fmt == "parquet" else "csv"


def concat_plate(outdir, objects, fmt, plate_name):
    """Merge the per-field tables into one table per object set.

    Field tables are named ``<well>_s<site>_<ObjectSet>.<ext>``; the ``_s*``
    glob selects only those (never the merged ``<plate>_<ObjectSet>.<ext>``).
    """
    ext = _ext(fmt)
    for obj in objects:
        merged = os.path.join(outdir, f"{plate_name}_{obj}.{ext}")
        files = sorted(
            f
            for f in glob.glob(os.path.join(outdir, f"*_s*_{obj}.{ext}"))
            if os.path.abspath(f) != os.path.abspath(merged)
        )
        if not files:
            print(f"  no {obj} field tables to merge", flush=True)
            continue
        if fmt == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.concat_tables([pq.read_table(f) for f in files]), merged)
        else:
            with open(merged, "w", newline="") as out:
                for i, f in enumerate(files):
                    with open(f) as inp:
                        if i:
                            next(inp, None)  # drop header on all but the first
                        out.writelines(inp)
        print(f"  merged {len(files)} {obj} tables -> {os.path.basename(merged)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", default=DEFAULT_IMAGES)
    ap.add_argument("--masks-dir", default=DEFAULT_MASKS)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--objects", default="Cells,Nuclei", help="comma list of object sets")
    ap.add_argument("--features", default=",".join(ALL_FEATURES),
                    help=f"comma list from {ALL_FEATURES}")
    ap.add_argument("--in_range", default="0.0,1.0",
                    help="input intensity range as a comma-separated list of two floats")
    ap.add_argument("--format", choices=["auto", "parquet", "csv"], default="auto")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N fields")
    ap.add_argument("--plate-name", default=None,
                    help="name for the merged plate tables (default: output dir basename)")
    ap.add_argument("--no-concat", dest="concat", action="store_false",
                    help="skip merging per-field tables into per-object-set plate tables")
    ap.add_argument("--concat-only", action="store_true",
                    help="only merge existing per-field tables, do not (re)process fields")
    ap.set_defaults(concat=True)
    args = ap.parse_args()

    objects = [o for o in args.objects.split(",") if o]
    features = [f for f in args.features.split(",") if f]
    in_range = [float(x) for x in args.in_range.split(",")]
    if len(in_range) != 2 or in_range[0] >= in_range[1]:
        print(f"Invalid --in_range {args.in_range} — must be two floats, min < max. Using default 0.0,1.0.", flush=True)
        in_range = (0.0, 1.0)

    fmt = resolve_format(args.format)
    os.makedirs(args.output_dir, exist_ok=True)
    plate_name = args.plate_name or os.path.basename(os.path.normpath(args.output_dir))

    if args.concat_only:
        print(f"Merging existing field tables in {args.output_dir} (plate '{plate_name}') ...", flush=True)
        concat_plate(args.output_dir, objects, fmt, plate_name)
        return

    base_objects = [o for o in objects if o in OBJECT_MASKS]
    fields = discover_fields(args.images_dir, args.masks_dir, base_objects, args.limit)
    if not fields:
        raise SystemExit("No fields discovered — check --images-dir / --masks-dir and the CONFIG patterns.")
    print(f"{len(fields)} fields | {args.workers} workers | format={fmt} | "
          f"objects={objects} | features={features}", flush=True)

    t0 = time.perf_counter()
    ok = err = nobj = 0
    tasks = [(f, args.output_dir, features, fmt, objects, in_range) for f in fields]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_field, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            fid, status, n, el, msg = fut.result()
            if status == "ok":
                ok += 1
                nobj += n
            else:
                err += 1
                print(f"  ERROR {fid}: {msg}", flush=True)
            if i % 10 == 0 or i == len(futures):
                rate = i / (time.perf_counter() - t0)
                eta = (len(futures) - i) / rate if rate else 0
                print(f"  [{i}/{len(futures)}] ok={ok} err={err} objects={nobj} "
                      f"{rate:.2f} fields/s  ETA {eta/60:.1f} min", flush=True)

    dt = time.perf_counter() - t0
    print(f"\nProcessed: {ok} ok, {err} errors, {nobj} objects in {dt/60:.1f} min "
          f"({len(fields)/dt:.2f} fields/s).", flush=True)

    if args.concat:
        print("Merging field tables into per-object-set plate tables ...", flush=True)
        concat_plate(args.output_dir, objects, fmt, plate_name)
    print(f"Output -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
