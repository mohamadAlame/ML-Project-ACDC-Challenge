"""Builds 'Group 26 Final Project Notebook.ipynb' from the project scripts.

Sections 1-7 run in a few minutes: they preprocess the data, load the saved trained models and
recompute the validation and test results from them. The full training code is in the appendix,
behind RUN_TRAINING, because training everything takes about 2.5 hours.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


def take(filename, names):
    """Copy functions/classes/constants from a script, unchanged, so the notebook uses the exact run code."""
    src = (ROOT / filename).read_text()
    out = []
    for node in ast.parse(src).body:
        name = getattr(node, "name", None) or (node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in names:
            out.append(ast.get_source_segment(src, node))
    return "\n\n\n".join(out)


cells = [
    md("""
# ACDC Cardiac MRI: Segmentation and Diagnosis
**CMPS 261, Group 26: Mohamad Alame, Sarra Fassakh**

The problem has two parts:
1. **Segmentation:** label every pixel of a heart MRI slice as right ventricle, myocardium (muscle wall), left ventricle or background.
2. **Diagnosis:** use measurements from those outlines to predict one of five patient groups.

Sections 1–7 run in a few minutes. They preprocess the data, load our saved trained models and recompute the results from them. The full training code is in the appendix, behind `RUN_TRAINING`, because training everything took about 2.5 hours.
"""),
    md("## 1. Setup"),
    code("""
import glob, json, os, re, sys, time
import numpy as np, pandas as pd, joblib, nibabel as nib, torch, torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import Image, display
from scipy import ndimage as ni
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.getcwd())
from acdc_forest_baseline import CROP, DATA, MODEL_DIR, OUT, SPACING, features, load_patient, scores

TEST_DATA = DATA.replace("training", "testing")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
RUN_TRAINING = False          # True reruns all training in the appendix (about 2.5 hours)
experiments = json.load(open(os.path.join(OUT, "experiments.json")))
print("device:", device, "| data found:", os.path.isdir(DATA), "| models found:", os.path.isdir(MODEL_DIR))
"""),
    md("""
## 2. Data loading and preprocessing

The scans are NIfTI files, read with `nibabel`. For every patient we take the two labelled moments (heart full and heart squeezed) and:
- **normalise brightness per scan**, clipping the extreme 1% and scaling to 0–1, because MRI brightness has no fixed scale;
- **split each 3D scan into 2D slices**;
- **resize to 1.5 mm per pixel**, so the same organ has the same size in pixels for every patient (nearest-neighbour for outlines, so labels stay valid);
- **crop or pad to 192 × 192**.

Patients, not slices, are split: 80 training, 20 validation (4 per group), and the official 50 test patients kept aside. The preprocessing code is `load_patient` in `acdc_forest_baseline.py` (appendix A).
"""),
    code("""
split = json.load(open(os.path.join(OUT, "split.json")))
t0 = time.time()
data, kept, total = {}, 0, 0
for n in split["train"] + split["val"]:
    s, k, t = load_patient(os.path.join(DATA, n))
    data[n], kept, total = s, kept + k, total + t
test = {os.path.basename(p): load_patient(p)[0] for p in sorted(glob.glob(glob.escape(TEST_DATA) + "/patient*"))}
tr = [s for n in split["train"] for s in data[n]]
va = [s for n in split["val"] for s in data[n]]
te = [s for n in sorted(test) for s in test[n]]
print(f"preprocessed in {time.time() - t0:.0f} s: {len(tr)} train, {len(va)} validation, {len(te)} test slices")
print(f"heart pixels kept after cropping: {kept / total:.2%}")

raw = [np.percentile(np.asarray(nib.load(f).dataobj), 99) for f in sorted(glob.glob(glob.escape(DATA) + "/patient*/*_frame??.nii.gz"))]
print(f"bright end (99th percentile) across raw scans: {min(raw):.0f} to {max(raw):.0f}  -> why we normalise")
"""),
    code("""
s = va[len(va) // 2]
fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
ax[0].imshow(s["img"].T, cmap="gray"); ax[0].set_title("preprocessed slice")
ax[1].imshow(s["img"].T, cmap="gray"); ax[1].imshow(np.ma.masked_where(s["mask"].T == 0, s["mask"].T), cmap="jet", alpha=.5)
ax[1].set_title("expert outline")
for a in ax: a.axis("off")
plt.show()
"""),
    md("""
## 3. Classical baseline: random forest

Each pixel becomes one row of 11 features (brightness, blurred brightness at four sizes, edge strength, brightness against surroundings, three position values, slice position). We sample up to 300 heart and 1,200 background pixels per slice. Tuning, in `acdc_forest_baseline.py` (appendix A), searched the background sampling, class weights, depth, leaf size, features per split and number of trees on the validation patients. Below are the saved tuning trials and final validation results.
"""),
    code("""
display(pd.read_csv(os.path.join(OUT, "tuning.csv"))[["stage", "bg", "class_weight", "depth", "leaf", "max_features", "trees", "macro_f1", "mean_precision", "mean_recall", "train_s"]])
fr = json.load(open(os.path.join(OUT, "forest_results.json")))
print("chosen settings:", fr["best_config"])
print("validation macro F1:", fr["validation"]["macro_f1"], "| 5-fold by patient:", fr["cv5_mean"], "±", fr["cv5_std"])
print("feature importance:", fr["feature_importance"])
print("without position features (validation macro F1):", experiments["forest_without_position"]["validation"]["macro_f1"])
"""),
    md("## 4. U-Net\n\nThe same U-Net design as in our experiments: it shrinks the slice three times to see context, then expands it back, reusing detail. Below we load the chosen saved model (E: augmentation + Dice loss, 80 epochs) and recompute its validation score."),
    code(take("acdc_experiments.py", {"confusion", "breakdown", "block", "UNet", "tensors", "predict"})),
    code("""
unet = UNet(16).to(device)
unet.load_state_dict(torch.load(os.path.join(MODEL_DIR, "acdc_unet_E.pt"), map_location=device))
print("parameters:", sum(p.numel() for p in unet.parameters()))
xv, yv = tensors(va)
print("validation macro F1 (recomputed):", scores(confusion(predict(unet, xv), yv))["macro_f1"])
display(pd.DataFrame([experiments["unet_pilot"]] + experiments["unet_variants"])[["name", "parameters", "epochs", "train_seconds", "val_macro_f1"]])
display(Image(os.path.join(OUT, "unet_variants_curves.png")))
"""),
    md("""
## 4b. Pretrained ResNet18-UNet (Sarra Fassakh)

A second deep model, to test whether transfer learning helps with only 100 training patients. The encoder is an ImageNet-pretrained ResNet18; the decoder is U-Net style. Training had two stages: first the pretrained encoder was frozen and only the decoder trained (learning rate 0.001, up to 10 epochs); then the last ResNet block (layer4) was unfrozen and fine-tuned with smaller learning rates (0.0001 for layer4, 0.0005 for the decoder, up to 15 epochs, early stopping). Same preprocessing, same patient split (`split.json`, seed 261) and the same 50 test patients as our U-Net. Trained on a laptop CPU in 39.2 minutes.

Its full code and outputs are in appendix E. It saves its weights as `resnet18_unet_acdc.pt` (48 MB). Below we load its saved results.
"""),
    code("""
rn = json.load(open(os.path.join(OUT, "resnet18_results.json")))
u_val = scores(confusion(predict(unet, xv), yv))
u_test = experiments["test_segmentation"]["U-Net"]["test"]
rows = []
for name, v, t, params in (("U-Net (from scratch)", u_val, u_test, 483204),
                           ("Pretrained ResNet18-UNet", None, None, rn["parameters"])):
    if v is None:
        vs, ts = rn["val_segmentation"], rn["test_segmentation"]
        rows.append({"model": name, "parameters": params, "val macro F1": vs["Macro_F1"], "test macro F1": ts["Macro_F1"],
                     "test RV F1": ts["RV_F1"], "test Myo F1": ts["Myo_F1"], "test LV F1": ts["LV_F1"]})
    else:
        rows.append({"model": name, "parameters": params, "val macro F1": v["macro_f1"], "test macro F1": t["macro_f1"],
                     "test RV F1": t["f1"][0], "test Myo F1": t["f1"][1], "test LV F1": t["f1"][2]})
display(pd.DataFrame(rows).round(3))
"""),
    md("## 5. Diagnosis models\n\nTwelve measurements per patient from the outlines: volumes, ejection fractions, muscle mass, body-size-adjusted values and the thickest wall. We compare logistic regression at five strengths of regularisation with 5-fold cross-validation repeated 5 times, on the 100 training patients' expert outlines."),
    code(take("acdc_experiments.py", {"meta", "measure", "MEASURES", "patient_table"})),
    code("""
names = sorted(split["train"] + split["val"])
Xd, yd = patient_table(names, DATA, [s for n in names for s in data[n]])
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=261)
for c in (0.01, 0.1, 1, 10, 100):
    sc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)), Xd, yd, cv=cv)
    print(f"logistic regression C={c}: accuracy {sc.mean():.3f} ± {sc.std():.3f}")
print("saved comparison, including the small neural network:")
display(pd.DataFrame(experiments["diagnosis_cv"]).T)
diag = joblib.load(os.path.join(MODEL_DIR, "acdc_diagnosis_final.joblib"))["model"]
"""),
    md("## 6. Final test on the 50 test patients\n\nUsed once, after every choice above was made on the validation patients. Here we recompute it from the saved models."),
    code(take("acdc_experiments.py", {"confidence"})),
    code("""
xe, ye = tensors(te)
u_prob = predict(unet, xe, probs=True)
u_pred = [p.argmax(0).numpy().reshape(-1) for p in u_prob]
u_seg = scores(sum(confusion(p, s["mask"]) for p, s in zip(u_pred, te)))
print("U-Net test:", u_seg)
print("U-Net confidence:", confidence([p.numpy() for p in u_prob], u_pred, te))

forest_file = os.path.join(MODEL_DIR, "acdc_forest.joblib")
if os.path.exists(forest_file):          # 272 MB, not on GitHub; rebuilt by acdc_forest_baseline.py
    forest = joblib.load(forest_file)
    f_pred = [forest.predict(features(s)) for s in te]
    print("Forest test:", scores(sum(confusion(p, s["mask"]) for p, s in zip(f_pred, te))))
"""),
    code("""
tnames = sorted(test)
X_true, y_test = patient_table(tnames, TEST_DATA, te)
X_unet, _ = patient_table(tnames, TEST_DATA, te, u_pred)
rows = [("expert outlines", X_true), ("U-Net outlines", X_unet)]
if os.path.exists(forest_file):
    rows.append(("forest outlines", patient_table(tnames, TEST_DATA, te, f_pred)[0]))
display(pd.DataFrame([{"outlines": k, "diagnosis accuracy": (diag.predict(X) == y_test).mean(),
                       "LV EF error (points)": 100 * np.abs(X[:, 2] - X_true[:, 2]).mean(),
                       "RV EF error (points)": 100 * np.abs(X[:, 5] - X_true[:, 5]).mean()} for k, X in rows]
                     # ResNet row: accuracy printed by acdc_resnet18_unet.ipynb (appendix E), errors from its results file
                     + [{"outlines": "ResNet18-UNet outlines", "diagnosis accuracy": 0.86,
                         "LV EF error (points)": rn["test_mae"]["lv_ef_points"], "RV EF error (points)": rn["test_mae"]["rv_ef_points"]}]).round(3))
"""),
    md("## 7. Failure and complexity analysis"),
    code("""
ub = breakdown(u_pred, te)
display(pd.DataFrame([{"subset": f"{k}: {sub}", "U-Net": v, "ResNet18-UNet": rn["test_breakdown"][k][sub]}
                      for k in ub for sub, v in ub[k].items()]).round(3))
uc = experiments["test_segmentation"]["U-Net"]["confidence"]
display(pd.DataFrame([{"model": "U-Net", **uc}, {"model": "ResNet18-UNet", **rn["test_confidence"]}]))
display(pd.DataFrame([
    {"model": "Random forest", "parameters": "200 trees", "train": "3 min per fit, ~1 h with tuning", "s per slice": experiments["test_segmentation"]["random forest"]["seconds_per_slice"], "file MB": 272, "test macro F1": experiments["test_segmentation"]["random forest"]["test"]["macro_f1"]},
    {"model": "U-Net (final)", "parameters": 483204, "train": "30 min", "s per slice": experiments["test_segmentation"]["U-Net"]["seconds_per_slice"], "file MB": 2, "test macro F1": experiments["test_segmentation"]["U-Net"]["test"]["macro_f1"]},
    {"model": "Pretrained ResNet18-UNet", "parameters": rn["parameters"], "train": "39.2 min (CPU)", "s per slice": rn["test_seconds_per_slice"], "file MB": 48, "test macro F1": rn["test_segmentation"]["Macro_F1"]},
]))
display(Image(os.path.join(OUT, "test_examples.png")))
"""),
    md("## Appendix: full training code\n\nThe exact scripts we ran, in order. Set `RUN_TRAINING = True` in section 1 to rerun them. A rerun gives slightly different numbers, because training uses random shuffling and augmentation."),
]
for f, title in (("acdc_forest_baseline.py", "A. Preprocessing and random forest"),
                 ("acdc_diagnosis_baseline.py", "B. Diagnosis baseline"),
                 ("acdc_unet_pilot.py", "C. U-Net pilot"),
                 ("acdc_experiments.py", "D. U-Net variants, diagnosis tuning and final test")):
    cells.append(md(f"### {title} (`{f}`)"))
    cells.append(md("```python\n" + (ROOT / f).read_text().rstrip() + "\n```"))
    cells.append(code(f"# Runs the script above, so the code is exactly what produced our results.\nif RUN_TRAINING:\n    %run {f}"))

cells.append(md("### E. Pretrained ResNet18-UNet (Sarra Fassakh)\n\nHer full code, cell by cell, followed by the outputs it produced when she ran it on her laptop (CPU, 39.2 min of training). Only one change from her original: the data, split and model paths point to this project's folders. Set `RUN_RESNET_TRAINING = True` to retrain it."))
cells.append(code("RUN_RESNET_TRAINING = False   # True retrains the ResNet18-UNet (about 40 min on a 6-core CPU)"))
PATHS = {'TRAIN_DIR  = "training"': 'TRAIN_DIR  = DATA',
         'TEST_DIR   = "testing"': 'TEST_DIR   = TEST_DATA',
         'SPLIT_PATH = "split.json"': 'SPLIT_PATH = os.path.join(OUT, "split.json")',
         'SAVE_PATH  = "resnet18_unet_acdc.pt"': 'SAVE_PATH  = os.path.join(MODEL_DIR, "resnet18_unet_acdc.pt")'}
for c in json.loads((ROOT / "acdc_resnet18_unet.ipynb").read_text())["cells"]:
    src = "".join(c["source"]).rstrip()
    if c["cell_type"] != "code":
        cells.append(md(src))
        continue
    for a, b in PATHS.items():
        src = src.replace(a, b)
    title = src.splitlines()[0] if src.startswith("#") else "# ResNet18-UNet"
    cells.append(code(title + "\nif RUN_RESNET_TRAINING:\n" + "\n".join("    " + l if l else "" for l in src.splitlines())))
    text = "".join("".join(o.get("text", "")) for o in c.get("outputs", []) if o.get("output_type") == "stream").rstrip()
    pngs = [o["data"]["image/png"] for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
    if text or pngs:
        cell = md("**Her output:**\n" + ("```text\n" + text + "\n```\n" if text else "")
                  + "".join(f"\n![plot](attachment:plot{k}.png)\n" for k in range(len(pngs))))
        if pngs:
            cell["attachments"] = {f"plot{k}.png": {"image/png": png if isinstance(png, str) else "".join(png)} for k, png in enumerate(pngs)}
        cells.append(cell)

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(ROOT / "Group 26 Final Project Notebook.ipynb").write_text(json.dumps(nb, indent=1))
print("notebook built")
