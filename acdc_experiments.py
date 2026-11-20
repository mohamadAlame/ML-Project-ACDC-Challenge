"""CMPS 261 project, Group 26. Final experiments on ACDC.

1. Forest without position features (does it just memorise where the heart usually is?)
2. U-Net variants: augmentation, Dice loss, wider network, longer training
3. Diagnosis models: tuned logistic regression, small neural network, random forest
4. Final test on the 50 untouched test patients: segmentation, heart measurements,
   diagnosis from predicted outlines, and whether model confidence can be trusted.

Model choices are made on the 20 validation patients only. The test set is used once, at the end.
Run: python acdc_experiments.py   (add --smoke for a 2-minute check)
"""
import csv, glob, json, os, re, sys, time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage as ni
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()))
from acdc_forest_baseline import (CLASSES, CROP, DATA, MODEL_DIR, OUT, SPACING, features, forest,
                                  load_patient, log, sample_rows, scores)

SMOKE = "--smoke" in sys.argv
TEST_DATA = DATA.replace("training", "testing")
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(261)
np.random.seed(261)
RESULTS = {}


def save():
    json.dump(RESULTS, open(os.path.join(OUT, "experiments" + ("_smoke" if SMOKE else "") + ".json"), "w"), indent=1)


def confusion(pred, true):
    return np.bincount((np.asarray(true).reshape(-1).astype(np.int64) * 4 + np.asarray(pred).reshape(-1)), minlength=16).reshape(4, 4)


def breakdown(preds, slices):
    """Macro F1 overall and by moment, slice position and group."""
    def sub(cond):
        i = [k for k, s in enumerate(slices) if cond(s)]
        return scores(sum(confusion(preds[k], slices[k]["mask"]) for k in i))["macro_f1"]
    third = lambda s: "first third" if s["z"] < 1 / 3 else ("middle" if s["z"] < 2 / 3 else "last third")
    return dict(by_moment={m: sub(lambda s, m=m: s["moment"] == m) for m in ("ED", "ES")},
                by_slice_position={p: sub(lambda s, p=p: third(s) == p) for p in ("first third", "middle", "last third")},
                by_group={g: sub(lambda s, g=g: s["group"] == g) for g in sorted({s["group"] for s in slices})})


# ---------------- data ----------------
split = json.load(open(os.path.join(OUT, "split.json")))
train_names, val_names = split["train"], split["val"]
test_paths = sorted(glob.glob(glob.escape(TEST_DATA) + "/patient*"))
if SMOKE:
    train_names, val_names, test_paths = train_names[::8], val_names[:3], test_paths[::17]
data = {n: load_patient(os.path.join(DATA, n))[0] for n in train_names + val_names}
test = {os.path.basename(p): load_patient(p)[0] for p in test_paths}
tr = [s for n in train_names for s in data[n]]
va = [s for n in val_names for s in data[n]]
te = [s for n in sorted(test) for s in test[n]]
log(f"{len(tr)} train, {len(va)} val, {len(te)} test slices; device {device}")


# ---------------- 1. forest without position features ----------------
NO_POS = [0, 1, 2, 3, 4, 5, 6, 10]          # drop y, x and distance from centre
best_forest = dict(fg=300, bg=1200, trees=200, depth=20, leaf=5, max_features="sqrt", class_weight=None)
if SMOKE:
    best_forest.update(trees=10, bg=200)
X, Y = sample_rows(tr, best_forest["fg"], best_forest["bg"], np.random.default_rng(261))
t0 = time.time()
m = forest(best_forest).fit(X[:, NO_POS], Y)
fit_s = time.time() - t0
cm = sum(confusion(m.predict(features(s)[:, NO_POS]), s["mask"]) for s in va)
RESULTS["forest_without_position"] = dict(validation=scores(cm), train_seconds=round(fit_s, 1),
                                          with_position_macro_f1=0.5901)
log("forest without position", RESULTS["forest_without_position"]["validation"])
del m, X, Y
save()


# ---------------- 2. U-Net variants ----------------
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())


class UNet(nn.Module):
    """Same design as the pilot: shrink 3 times to see context, expand back to reuse detail."""

    def __init__(self, b=16):
        super().__init__()
        self.d1, self.d2, self.d3 = block(1, b), block(b, 2 * b), block(2 * b, 4 * b)
        self.mid = block(4 * b, 8 * b)
        self.u3, self.c3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), block(8 * b, 4 * b)
        self.u2, self.c2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), block(4 * b, 2 * b)
        self.u1, self.c1 = nn.ConvTranspose2d(2 * b, b, 2, 2), block(2 * b, b)
        self.out, self.pool = nn.Conv2d(b, 4, 1), nn.MaxPool2d(2)

    def forward(self, x):
        a = self.d1(x)
        c = self.d2(self.pool(a))
        d = self.d3(self.pool(c))
        m = self.mid(self.pool(d))
        d = self.c3(torch.cat([self.u3(m), d], 1))
        c = self.c2(torch.cat([self.u2(d), c], 1))
        return self.out(self.c1(torch.cat([self.u1(c), a], 1)))


def tensors(slices):
    return (torch.tensor(np.stack([s["img"] for s in slices]))[:, None].float(),
            torch.tensor(np.stack([s["mask"] for s in slices])).long())


def augment(x, y):
    """Random small rotation (±15°), shift (±10%) and zoom (0.9–1.1), same for image and outline.
    Real scans are not always centred or aligned the same way, so the model should not rely on that."""
    n = len(x)
    a = (torch.rand(n) * 2 - 1) * np.pi / 12
    s = 0.9 + torch.rand(n) * 0.2
    t = (torch.rand(n, 2) * 2 - 1) * 0.1
    theta = torch.stack([torch.stack([torch.cos(a) / s, -torch.sin(a) / s, t[:, 0]], 1),
                         torch.stack([torch.sin(a) / s, torch.cos(a) / s, t[:, 1]], 1)], 1)
    grid = Fn.affine_grid(theta, x.shape, align_corners=False)
    x = Fn.grid_sample(x, grid, mode="bilinear", align_corners=False)
    y = Fn.grid_sample(y[:, None].float(), grid, mode="nearest", align_corners=False)[:, 0].long()
    return x, y


def dice_loss(logits, y):
    """1 − average overlap of the three heart structures. Rewards overlap directly,
    which matters because each structure is only about 1% of the pixels."""
    p = logits.softmax(1)[:, 1:]
    t = Fn.one_hot(y, 4).permute(0, 3, 1, 2)[:, 1:].float()
    inter = (p * t).sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (p.sum((0, 2, 3)) + t.sum((0, 2, 3)) + 1)).mean()


def predict(model, x, probs=False):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            p = model(x[i:i + 32].to(device)).softmax(1).cpu()
            out.append(p if probs else p.argmax(1))
    return torch.cat(out)


xt, yt = tensors(tr)
xv, yv = tensors(va)
xe, ye = tensors(te)
VARIANTS = [
    dict(name="B: + augmentation", base=16, aug=True, dice=False, epochs=40),
    dict(name="C: + augmentation + Dice loss", base=16, aug=True, dice=True, epochs=40),
    dict(name="D: wider (32) + aug + Dice", base=32, aug=True, dice=True, epochs=40),
    dict(name="E: aug + Dice, 80 epochs", base=16, aug=True, dice=True, epochs=80),
]
RESULTS["unet_pilot"] = dict(name="A: pilot (no augmentation, cross-entropy)", base=16, epochs=30,
                             val_macro_f1=0.9017, parameters=483204, train_seconds=617.8)
RESULTS["unet_variants"] = []
curves = {}
for v in VARIANTS:
    if SMOKE:
        v = {**v, "epochs": 1}
    torch.manual_seed(261)
    model = UNet(v["base"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, hist, start = -1, [], time.time()
    path = os.path.join(MODEL_DIR, f"acdc_unet_{v['name'][0]}{'_smoke' if SMOKE else ''}.pt")
    for epoch in range(1, v["epochs"] + 1):
        model.train()
        order = torch.randperm(len(xt))
        for i in range(0, len(xt), 16):
            xb, yb = xt[order[i:i + 16]], yt[order[i:i + 16]]
            if v["aug"]:
                xb, yb = augment(xb, yb)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = Fn.cross_entropy(out, yb) + (dice_loss(out, yb) if v["dice"] else 0)
            loss.backward()
            opt.step()
        f1 = scores(confusion(predict(model, xv), yv))["macro_f1"]
        hist.append(f1)
        if f1 > best:
            best, best_epoch = f1, epoch
            torch.save(model.state_dict(), path)
    train_s = time.time() - start
    curves[v["name"]] = hist
    row = dict(name=v["name"], base=v["base"], augmentation=v["aug"], dice=v["dice"], epochs=v["epochs"],
               parameters=sum(p.numel() for p in model.parameters()), train_seconds=round(train_s, 1),
               val_macro_f1=best, best_epoch=best_epoch,
               last5_mean=round(float(np.mean(hist[-5:])), 4), last5_min=round(float(np.min(hist[-5:])), 4),
               file_mb=round(os.path.getsize(path) / 1e6, 2), path=path)
    RESULTS["unet_variants"].append(row)
    log(row)
    save()

pick = max(RESULTS["unet_variants"], key=lambda r: r["val_macro_f1"])
RESULTS["unet_chosen"] = pick["name"]
model = UNet(pick["base"]).to(device)
model.load_state_dict(torch.load(pick["path"]))
fig, ax = plt.subplots(figsize=(7, 4))
for k, h in curves.items():
    ax.plot(range(1, len(h) + 1), h, label=k)
ax.set_xlabel("epoch"); ax.set_ylabel("validation macro F1"); ax.set_ylim(0.6, 0.95); ax.legend(fontsize=8)
fig.savefig(os.path.join(OUT, "unet_variants_curves.png"), dpi=120, bbox_inches="tight")


# ---------------- 3. heart measurements and diagnosis models ----------------
def meta(path):
    info = dict(re.findall(r"(\w+): (\S+)", open(os.path.join(path, "Info.cfg")).read()))
    z = float(nib.load(glob.glob(glob.escape(path) + "/*_frame??.nii.gz")[0]).header.get_zooms()[2])
    return info, z


def measure(ed, es, z, h, w):
    """12 measurements from outline stacks (slices x 192 x 192, at 1.5 mm pixels)."""
    ml = SPACING * SPACING * z / 1000
    lv_ed, lv_es = (ed == 3).sum() * ml, (es == 3).sum() * ml
    rv_ed, rv_es = (ed == 1).sum() * ml, (es == 1).sum() * ml
    mass = (ed == 2).sum() * ml * 1.05
    bsa = np.sqrt(h * w / 3600)
    thick = max(2 * ni.distance_transform_edt(sl == 2, sampling=SPACING).max() for sl in ed)
    lv_ef = (lv_ed - lv_es) / max(lv_ed, 1e-6)
    rv_ef = (rv_ed - rv_es) / max(rv_ed, 1e-6)
    return [lv_ed, lv_es, lv_ef, rv_ed, rv_es, rv_ef, mass, lv_ed / bsa, rv_ed / bsa, mass / bsa,
            rv_ed / max(lv_ed, 1e-6), thick]


MEASURES = ["lv_edv", "lv_esv", "lv_ef", "rv_edv", "rv_esv", "rv_ef", "muscle_mass", "lv_edv_per_m2",
            "rv_edv_per_m2", "mass_per_m2", "rv_lv_ratio", "max_thickness"]


def patient_table(names, root, slices, preds=None):
    """One row of measurements per patient, from expert outlines or from predictions."""
    rows, groups = [], []
    for n in names:
        info, z = meta(os.path.join(root, n))
        idx = {m: [i for i, s in enumerate(slices) if s["patient"] == n and s["moment"] == m] for m in ("ED", "ES")}
        get = (lambda i: slices[i]["mask"]) if preds is None else (lambda i: np.asarray(preds[i]).reshape(CROP, CROP))
        ed = np.stack([get(i) for i in idx["ED"]])
        es = np.stack([get(i) for i in idx["ES"]])
        rows.append(measure(ed, es, z, float(info["Height"]), float(info["Weight"])))
        groups.append(info.get("Group", "?"))
    return np.array(rows), np.array(groups)


all_train = sorted(train_names + val_names)
everything = [s for n in all_train for s in data[n]]
Xd, yd = patient_table(all_train, DATA, everything)
cv = RepeatedStratifiedKFold(n_splits=5 if not SMOKE else 2, n_repeats=5 if not SMOKE else 1, random_state=261)
candidates = {f"logistic regression C={c}": make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
              for c in (0.01, 0.1, 1, 10, 100)}
candidates.update({f"neural network (16 hidden) alpha={a}": make_pipeline(StandardScaler(), MLPClassifier((16,), alpha=a, max_iter=5000, random_state=261))
                   for a in (0.001, 0.01, 0.1, 1)})
candidates["random forest"] = RandomForestClassifier(n_estimators=300, random_state=261, n_jobs=-1)
RESULTS["diagnosis_cv"] = {}
for k, mdl in candidates.items():
    sc = cross_val_score(mdl, Xd, yd, cv=cv, scoring="accuracy")
    RESULTS["diagnosis_cv"][k] = dict(accuracy=round(float(sc.mean()), 4), std=round(float(sc.std()), 4))
    log(k, RESULTS["diagnosis_cv"][k])
best_lr = max((k for k in candidates if k.startswith("logistic")), key=lambda k: RESULTS["diagnosis_cv"][k]["accuracy"])
best_nn = max((k for k in candidates if k.startswith("neural")), key=lambda k: RESULTS["diagnosis_cv"][k]["accuracy"])
RESULTS["diagnosis_chosen"] = dict(classical=best_lr, neural=best_nn)
clf = {k: candidates[k].fit(Xd, yd) for k in (best_lr, best_nn)}
joblib.dump(dict(model=clf[best_lr], features=MEASURES), os.path.join(MODEL_DIR, "acdc_diagnosis_final.joblib"))
save()


# ---------------- 4. final test (used once) ----------------
log("final test")
forest_model = joblib.load(os.path.join(MODEL_DIR, "acdc_forest.joblib"))
t0 = time.time()
f_prob = [forest_model.predict_proba(features(s)) for s in te]
forest_s = time.time() - t0
f_pred = [p.argmax(1) for p in f_prob]
t0 = time.time()
u_prob = predict(model, xe, probs=True)
unet_s = time.time() - t0
u_pred = [p.argmax(0).numpy().reshape(-1) for p in u_prob]


def confidence(probs, preds, slices):
    """On pixels the model calls heart: is its confidence close to how often it is right?"""
    conf, right = [], []
    for p, q, s in zip(probs, preds, slices):
        p = np.asarray(p).reshape(4, -1).T if p.ndim == 3 else np.asarray(p)
        keep = q > 0
        conf.append(p.max(1)[keep])
        right.append((q == s["mask"].reshape(-1))[keep])
    conf, right = np.concatenate(conf), np.concatenate(right)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(conf[bins == b].mean() - right[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return dict(calibration_error=round(float(ece), 4), mean_conf_when_right=round(float(conf[right].mean()), 4),
                mean_conf_when_wrong=round(float(conf[~right].mean()), 4) if (~right).any() else None,
                share_right=round(float(right.mean()), 4))


seg = {}
for name, preds, probs, secs in (("random forest", f_pred, f_prob, forest_s), ("U-Net", u_pred, [p.numpy() for p in u_prob], unet_s)):
    cm = sum(confusion(p, s["mask"]) for p, s in zip(preds, te))
    seg[name] = dict(test=scores(cm), seconds_per_slice=round(secs / len(te), 4), **breakdown(preds, te),
                     confidence=confidence(probs, preds, te))
    log(name, seg[name]["test"])
seg["all_background_accuracy"] = round(float(cm[0].sum() / cm.sum()), 4)
RESULTS["test_segmentation"] = seg

names = sorted(test)
X_true, y_test = patient_table(names, TEST_DATA, te)
X_unet, _ = patient_table(names, TEST_DATA, te, u_pred)
X_forest, _ = patient_table(names, TEST_DATA, te, f_pred)
diag = {}
for k, c in clf.items():
    diag[k] = {src: round(float((c.predict(Xs) == y_test).mean()), 4)
               for src, Xs in (("expert outlines", X_true), ("U-Net outlines", X_unet), ("forest outlines", X_forest))}
    log(k, diag[k])
err = lambda Xs, j: round(float(np.mean(np.abs(Xs[:, j] - X_true[:, j]))), 3)
RESULTS["test_diagnosis"] = dict(accuracy=diag, patients=len(names),
    mean_abs_error={src: dict(lv_ef_points=round(100 * err(Xs, 2), 1), rv_ef_points=round(100 * err(Xs, 5), 1),
                              lv_edv_ml=err(Xs, 0), rv_edv_ml=err(Xs, 3))
                    for src, Xs in (("U-Net outlines", X_unet), ("forest outlines", X_forest))})
save()

# Figure: 3 test patients, middle slice at the full moment: MRI, expert, forest, U-Net.
pp = {n: scores(sum(confusion(u_pred[i], te[i]["mask"]) for i, s in enumerate(te) if s["patient"] == n))["macro_f1"] for n in names}
order = sorted(names, key=pp.get)
colours = ListedColormap(["#e4572e", "#f2c14e", "#2e86ab"])
fig, ax = plt.subplots(3, 4, figsize=(13, 10.5))
for r, n in enumerate([order[0], order[len(order) // 2], order[-1]]):
    idx = [i for i, s in enumerate(te) if s["patient"] == n and s["moment"] == "ED"]
    i = idx[len(idx) // 2]
    for c, (lab, title) in enumerate(((None, f"{n} ({te[i]['group']})"), (te[i]["mask"], "Expert"),
                                      (f_pred[i].reshape(CROP, CROP), "Random forest"), (u_pred[i].reshape(CROP, CROP), f"U-Net (patient F1 {pp[n]:.2f})"))):
        ax[r, c].imshow(te[i]["img"].T, cmap="gray")
        if lab is not None:
            ax[r, c].imshow(np.ma.masked_where(lab.T == 0, lab.T), cmap=colours, vmin=1, vmax=3, alpha=.6, interpolation="nearest")
        ax[r, c].set_title(title, fontsize=10)
        ax[r, c].axis("off")
fig.legend(handles=[Patch(color=colours(i), label=CLASSES[i + 1]) for i in range(3)], loc="lower center", ncol=3)
fig.savefig(os.path.join(OUT, "test_examples.png"), dpi=110, bbox_inches="tight")
log("DONE")
