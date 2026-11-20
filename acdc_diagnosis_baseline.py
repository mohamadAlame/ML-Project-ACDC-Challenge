"""CMPS 261 project, Group 26. ACDC diagnosis from heart measurements.

One row per patient: volumes, ejection fraction, muscle mass and wall thickness, measured
from the EXPERT outlines. This is an upper bound: the real pipeline would measure from
predicted outlines. Labels come from Info.cfg. The testing folder is never read.

Run: python acdc_diagnosis_baseline.py
"""
import glob, json, os, re, time, warnings

import joblib
import nibabel as nib
import numpy as np
from scipy import ndimage as ni
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
WORK = os.environ.get("ACDC_WORK", os.path.join((os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()), "cmps261-work"))
DATA = os.path.join(WORK, "data/acdc/extracted/database/training")
OUT = os.path.join((os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()), "results")
RV, MUSCLE, LV = 1, 2, 3
GROUPS = ["NOR", "MINF", "DCM", "HCM", "RV"]


def measure(path):
    info = dict(re.findall(r"(\w+): (\S+)", open(path + "/Info.cfg").read()))
    name = os.path.basename(path)

    def scan(frame):
        n = nib.load(f"{path}/{name}_frame{int(frame):02d}_gt.nii.gz")
        return np.asarray(n.dataobj).astype(np.uint8), tuple(float(z) for z in n.header.get_zooms())

    ed, z = scan(info["ED"])
    es, _ = scan(info["ES"])
    ml = z[0] * z[1] * z[2] / 1000                      # one voxel, in mL
    lv_ed, lv_es = (ed == LV).sum() * ml, (es == LV).sum() * ml
    rv_ed, rv_es = (ed == RV).sum() * ml, (es == RV).sum() * ml
    mass = (ed == MUSCLE).sum() * ml * 1.05             # heart muscle is about 1.05 g/mL
    h, w = float(info["Height"]), float(info["Weight"])
    bsa = np.sqrt(h * w / 3600)                         # Mosteller body surface area, m2
    thick = max(2 * ni.distance_transform_edt(ed[:, :, k] == MUSCLE, sampling=z[:2]).max()
                for k in range(ed.shape[2]))            # thickest wall, roughly, in mm
    row = dict(lv_edv=lv_ed, lv_esv=lv_es, lv_ef=(lv_ed - lv_es) / lv_ed,
               rv_edv=rv_ed, rv_esv=rv_es, rv_ef=(rv_ed - rv_es) / rv_ed,
               muscle_mass=mass, lv_edv_per_m2=lv_ed / bsa, rv_edv_per_m2=rv_ed / bsa,
               mass_per_m2=mass / bsa, rv_lv_ratio=rv_ed / lv_ed, max_thickness=thick)
    return row, info["Group"], name


rows, y, names = zip(*[measure(p) for p in sorted(glob.glob(glob.escape(DATA) + "/patient*"))])
cols = list(rows[0])
X = np.array([[r[c] for c in cols] for r in rows])
y = np.array(y)

models = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "decision tree": DecisionTreeClassifier(max_depth=4, random_state=261),
    "random forest": RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=261),
}
results = {}
for name, m in models.items():
    t, accs, f1s, cm = time.time(), [], [], np.zeros((5, 5), int)
    for a, b in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=261).split(X, y):
        p = m.fit(X[a], y[a]).predict(X[b])
        accs.append((p == y[b]).mean())
        f1s.append(f1_score(y[b], p, average="macro"))
        cm += confusion_matrix(y[b], p, labels=GROUPS)
    results[name] = dict(accuracy=round(float(np.mean(accs)), 4), accuracy_std=round(float(np.std(accs)), 4),
                         macro_f1=round(float(np.mean(f1s)), 4), seconds=round(time.time() - t, 1),
                         confusion_matrix=cm.tolist())
    print(name, {k: v for k, v in results[name].items() if k != "confusion_matrix"}, flush=True)
    print(cm, flush=True)

imp = RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=261).fit(X, y).feature_importances_
results["feature_importance"] = dict(sorted(zip(cols, imp.round(4).tolist()), key=lambda kv: -kv[1]))
results["group_order"] = GROUPS
print(results["feature_importance"])
os.makedirs(OUT, exist_ok=True)
json.dump(results, open(OUT + "/diagnosis_results.json", "w"), indent=1)

# Save the logistic regression, fitted on all 100 training patients.
final = models["logistic regression"].fit(X, y)
os.makedirs(os.path.join(WORK, "models"), exist_ok=True)
joblib.dump(dict(model=final, features=cols), os.path.join(WORK, "models/acdc_diagnosis_logreg.joblib"))
