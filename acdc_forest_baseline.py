"""CMPS 261 project, Group 26. ACDC heart segmentation: random forest baseline.

Each pixel of a 2D MRI slice is one row of a table, and the forest labels it as
background, right ventricle, muscle wall or left ventricle.

Run:  python acdc_forest_baseline.py [--prep-only]
Data: cmps261-work/data/acdc/extracted/database/training (the testing folder is never read)
Out:  results/ next to this file. Model files: cmps261-work/models/
"""
import csv, glob, json, os, re, sys, time, warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage as ni
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ---- settings ----
WORK = os.environ.get("ACDC_WORK", os.path.join((os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()), "cmps261-work"))   # data and models live here
DATA = os.path.join(WORK, "data/acdc/extracted/database/training")
MODEL_DIR = os.path.join(WORK, "models")
OUT = os.path.join((os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()), "results")
SEED = 261
SPACING = 1.5        # mm per pixel after resampling
CROP = 192           # slice size after centre crop or pad, in pixels
VAL_PER_GROUP = 4    # 4 x 5 groups = 20 validation patients
CLASSES = ["background", "right ventricle", "muscle wall", "left ventricle"]
FEATURES = ["brightness", "blur s=1", "blur s=2", "blur s=4", "blur s=8", "edge strength",
            "brightness vs surroundings", "y position", "x position", "distance from centre",
            "slice position"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---- preprocessing ----
def crop_or_pad(a, size):
    """Centre crop or zero-pad a 2D array to size x size."""
    h, w = a.shape
    y0, x0 = max((h - size) // 2, 0), max((w - size) // 2, 0)
    c = a[y0:y0 + size, x0:x0 + size]
    out = np.zeros((size, size), a.dtype)
    oy, ox = (size - c.shape[0]) // 2, (size - c.shape[1]) // 2
    out[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return out


def load_patient(path, size=CROP):
    """Return the patient's labelled slices as a list of dicts, plus heart pixels kept and total."""
    slices, kept, total = [], 0, 0
    info = dict(re.findall(r"(\w+): (\S+)", open(os.path.join(path, "Info.cfg")).read()))
    name = os.path.basename(path)
    for moment, frame in (("ED", int(info["ED"])), ("ES", int(info["ES"]))):
        f = os.path.join(path, f"{name}_frame{frame:02d}.nii.gz")
        img = nib.load(f)
        vol = np.asarray(img.dataobj, dtype=np.float32)
        gt = np.asarray(nib.load(f.replace(".nii.gz", "_gt.nii.gz")).dataobj).astype(np.uint8)
        sy, sx = (float(s) for s in img.header.get_zooms()[:2])
        lo, hi = np.percentile(vol, [1, 99])          # brightness: clip, then scale to 0-1, per scan
        vol = np.clip((vol - lo) / (hi - lo), 0, 1)
        n = vol.shape[2]
        for k in range(n):
            a = ni.zoom(vol[:, :, k], (sy / SPACING, sx / SPACING), order=1)   # linear for images
            b = ni.zoom(gt[:, :, k], (sy / SPACING, sx / SPACING), order=0)    # nearest for labels
            b2 = crop_or_pad(b, size)
            total += int((b > 0).sum())
            kept += int((b2 > 0).sum())
            slices.append(dict(img=crop_or_pad(a, size), mask=b2, z=k / max(n - 1, 1),
                               patient=name, group=info["Group"], moment=moment, k=k))
    return slices, kept, total


GY, GX = np.mgrid[:CROP, :CROP]
GY, GX = ((GY - CROP / 2) / CROP).astype(np.float32), ((GX - CROP / 2) / CROP).astype(np.float32)
GR = np.hypot(GY, GX)


def features(s):
    """One row of 11 numbers per pixel."""
    a = s["img"]
    blur = [ni.gaussian_filter(a, g) for g in (1, 2, 4, 8)]
    cols = [a, *blur, ni.gaussian_gradient_magnitude(a, 2), a - blur[3], GY, GX, GR,
            np.full_like(a, s["z"])]
    return np.stack(cols, -1).reshape(-1, len(cols)).astype(np.float32)


def sample_rows(slices, fg_n, bg_n, rng):
    """Training rows: up to fg_n heart pixels and bg_n background pixels per slice."""
    X, Y = [], []
    for s in slices:
        F, l = features(s), s["mask"].ravel()
        fg, bg = np.flatnonzero(l > 0), np.flatnonzero(l == 0)
        fg = rng.choice(fg, min(len(fg), fg_n), replace=False) if len(fg) else fg
        bg = rng.choice(bg, min(len(bg), bg_n), replace=False)
        i = np.r_[fg, bg]
        X.append(F[i])
        Y.append(l[i])
    return np.vstack(X), np.concatenate(Y)


# ---- evaluation ----
def scores(cm):
    d = np.diag(cm).astype(float)
    p, r = d / np.maximum(cm.sum(0), 1), d / np.maximum(cm.sum(1), 1)
    f = 2 * p * r / np.maximum(p + r, 1e-9)
    return dict(precision=p[1:].round(4).tolist(), recall=r[1:].round(4).tolist(),
                f1=f[1:].round(4).tolist(), macro_f1=round(float(f[1:].mean()), 4),
                pixel_accuracy=round(float(d.sum() / cm.sum()), 4))


def predict_all(model, feats):
    return [model.predict(F).astype(np.uint8) for F in feats]


def evaluate(preds, slices):
    cm = np.zeros((4, 4), np.int64)
    for p, s in zip(preds, slices):
        cm += confusion_matrix(s["mask"].ravel(), p, labels=[0, 1, 2, 3])
    return cm


def forest(cfg):
    return RandomForestClassifier(n_estimators=cfg["trees"], max_depth=cfg["depth"],
                                  min_samples_leaf=cfg["leaf"], max_features=cfg["max_features"],
                                  class_weight=cfg["class_weight"], n_jobs=-1, random_state=SEED)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    patients = sorted(glob.glob(DATA + "/patient*"))

    t = time.time()
    data, kept, total = {}, 0, 0
    for p in patients:
        s, k, n = load_patient(p)
        data[os.path.basename(p)], kept, total = s, kept + k, total + n
    prep_s = time.time() - t
    log(f"preprocessed {sum(map(len, data.values()))} slices from {len(data)} patients in {prep_s:.0f}s")
    log(f"heart pixels kept after crop: {kept}/{total} = {kept / total:.5f}")
    if "--prep-only" in sys.argv:
        return

    names = sorted(data)
    groups = {n: data[n][0]["group"] for n in names}
    val = sorted(v for g in sorted(set(groups.values()))
                 for v in rng.choice([n for n in names if groups[n] == g], VAL_PER_GROUP, replace=False))
    train = [n for n in names if n not in val]
    json.dump(dict(train=train, val=val), open(os.path.join(OUT, "split.json"), "w"), indent=1)
    tr = [s for n in train for s in data[n]]
    va = [s for n in val for s in data[n]]
    va_feats = [features(s) for s in va]
    log(f"split: {len(train)} train patients ({len(tr)} slices), {len(val)} val patients ({len(va)} slices)")

    rows = []

    def trial(stage, cfg, X=None, Y=None):
        if X is None:
            X, Y = sample_rows(tr, cfg["fg"], cfg["bg"], np.random.default_rng(SEED))
        t0 = time.time()
        m = forest(cfg).fit(X, Y)
        fit_s = time.time() - t0
        t0 = time.time()
        sc = scores(evaluate(predict_all(m, va_feats), va))
        pred_s = time.time() - t0
        row = dict(stage=stage, **cfg, rows=len(Y), train_s=round(fit_s, 1), predict_s=round(pred_s, 1),
                   macro_f1=sc["macro_f1"], f1_rv=sc["f1"][0], f1_muscle=sc["f1"][1], f1_lv=sc["f1"][2],
                   mean_precision=round(float(np.mean(sc["precision"])), 4),
                   mean_recall=round(float(np.mean(sc["recall"])), 4),
                   pixel_accuracy=sc["pixel_accuracy"])
        rows.append(row)
        log(row)
        with open(os.path.join(OUT, "tuning.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return row

    # Stage 1: how many background pixels to train on, and whether to reweight classes.
    best = dict(fg=300, bg=300, trees=100, depth=20, leaf=5, max_features="sqrt", class_weight="balanced")
    results = [trial("1 sampling", {**best, "bg": bg, "class_weight": cw})
               for bg in (300, 1200, 3000) for cw in ("balanced", None)]
    top = max(results, key=lambda r: r["macro_f1"])
    best.update(bg=top["bg"], class_weight=top["class_weight"])

    # Stage 2: forest settings, one at a time, keeping any improvement.
    X, Y = sample_rows(tr, best["fg"], best["bg"], np.random.default_rng(SEED))
    score = top["macro_f1"]
    for key, values in (("depth", (12, 30)), ("leaf", (1, 20)), ("max_features", (0.5,)), ("trees", (200,))):
        for v in values:
            r = trial("2 forest", {**best, key: v}, X, Y)
            if r["macro_f1"] > score:
                score = r["macro_f1"]
                best[key] = v
    log("best config", best)

    # Final model: refit on the training patients, full validation report.
    t0 = time.time()
    model = forest(best).fit(X, Y)
    fit_s = time.time() - t0
    t0 = time.time()
    preds = predict_all(model, va_feats)
    pred_s = time.time() - t0
    cm = evaluate(preds, va)
    joblib.dump(model, os.path.join(MODEL_DIR, "acdc_forest.joblib"), compress=3)

    def sub(cond):
        idx = [i for i, s in enumerate(va) if cond(s)]
        return scores(evaluate([preds[i] for i in idx], [va[i] for i in idx]))

    third = lambda s: "top" if s["z"] < 1 / 3 else ("middle" if s["z"] < 2 / 3 else "bottom")
    per_patient = {n: sub(lambda s, n=n: s["patient"] == n)["macro_f1"] for n in val}
    report = dict(
        best_config=best, training_rows=len(Y), train_seconds=round(fit_s, 1),
        predict_seconds_val=round(pred_s, 1), preprocess_seconds=round(prep_s, 1),
        heart_pixels_kept=round(kept / total, 5), validation=scores(cm), confusion_matrix=cm.tolist(),
        all_background_accuracy=round(float(cm[0].sum() / cm.sum()), 4),
        by_group={g: sub(lambda s, g=g: s["group"] == g)["macro_f1"] for g in sorted(set(groups.values()))},
        by_moment={m: sub(lambda s, m=m: s["moment"] == m)["macro_f1"] for m in ("ED", "ES")},
        by_slice_position={p: sub(lambda s, p=p: third(s) == p)["macro_f1"] for p in ("top", "middle", "bottom")},
        per_patient=per_patient,
        feature_importance=dict(sorted(zip(FEATURES, model.feature_importances_.round(4).tolist()),
                                       key=lambda kv: -kv[1])),
    )

    # Split sensitivity: 5-fold by patient with the chosen settings.
    cv = []
    y = [groups[n] for n in names]
    for i, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(names, y)):
        ftr = [s for j in a for s in data[names[j]]]
        fva = [s for j in b for s in data[names[j]]]
        Xf, Yf = sample_rows(ftr, best["fg"], best["bg"], np.random.default_rng(SEED))
        m = forest(best).fit(Xf, Yf)
        c = scores(evaluate([m.predict(features(s)).astype(np.uint8) for s in fva], fva))["macro_f1"]
        cv.append(c)
        log(f"fold {i + 1}: macro F1 {c}")
    report["cv5_macro_f1"] = cv
    report["cv5_mean"] = round(float(np.mean(cv)), 4)
    report["cv5_std"] = round(float(np.std(cv)), 4)
    json.dump(report, open(os.path.join(OUT, "forest_results.json"), "w"), indent=1)

    # Figure: worst, median and best validation patients, middle slice at the full moment.
    order = sorted(val, key=per_patient.get)
    pick = [order[0], order[len(order) // 2], order[-1]]
    colours = ListedColormap(["#e4572e", "#f2c14e", "#2e86ab"])
    fig, ax = plt.subplots(3, 3, figsize=(10, 10.5))
    for r, n in enumerate(pick):
        idx = [i for i, s in enumerate(va) if s["patient"] == n and s["moment"] == "ED"]
        i = idx[len(idx) // 2]
        pred = preds[i].reshape(CROP, CROP)
        for c, (lab, title) in enumerate(((None, "MRI slice"), (va[i]["mask"], "Expert labels"), (pred, "Forest prediction"))):
            ax[r, c].imshow(va[i]["img"].T, cmap="gray")
            if lab is not None:
                ax[r, c].imshow(np.ma.masked_where(lab.T == 0, lab.T), cmap=colours, vmin=1, vmax=3,
                                alpha=.6, interpolation="nearest")
            ax[r, c].set_title(f"{title}" + (f"\n{n} ({va[i]['group']}), patient macro F1 {per_patient[n]:.2f}" if c == 0 else ""), fontsize=10)
            ax[r, c].axis("off")
    fig.legend(handles=[Patch(color=colours(i), label=CLASSES[i + 1]) for i in range(3)], loc="lower center", ncol=3)
    fig.savefig(os.path.join(OUT, "forest_examples.png"), dpi=120, bbox_inches="tight")
    log("DONE", report["validation"], "cv", report["cv5_mean"], report["cv5_std"])


if __name__ == "__main__":
    main()
