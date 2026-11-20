"""CMPS 261 project, Group 26. ACDC heart segmentation: small U-Net pilot.

Same preprocessing, patient split and scoring as acdc_forest_baseline.py, so the two
models are directly comparable. Whole slices go in; a label per pixel comes out.
This is a first, untuned run. The testing folder is never read.

Run: python acdc_unet_pilot.py
"""
import glob, json, os, sys, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()))
from acdc_forest_baseline import CLASSES, CROP, DATA, MODEL_DIR, OUT, load_patient, log, scores

EPOCHS, BATCH, LR, BASE = 30, 16, 1e-3, 16
torch.manual_seed(261)
np.random.seed(261)
device = "mps" if torch.backends.mps.is_available() else "cpu"


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())


class UNet(nn.Module):
    """Shrinks the image 3 times to see context, then expands it back, reusing detail."""

    def __init__(self, b=BASE):
        super().__init__()
        self.d1, self.d2, self.d3 = block(1, b), block(b, 2 * b), block(2 * b, 4 * b)
        self.mid = block(4 * b, 8 * b)
        self.u3, self.c3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), block(8 * b, 4 * b)
        self.u2, self.c2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), block(4 * b, 2 * b)
        self.u1, self.c1 = nn.ConvTranspose2d(2 * b, b, 2, 2), block(2 * b, b)
        self.out = nn.Conv2d(b, 4, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        a = self.d1(x)
        c = self.d2(self.pool(a))
        d = self.d3(self.pool(c))
        m = self.mid(self.pool(d))
        d = self.c3(torch.cat([self.u3(m), d], 1))
        c = self.c2(torch.cat([self.u2(d), c], 1))
        a = self.c1(torch.cat([self.u1(c), a], 1))
        return self.out(a)


def tensors(slices):
    x = torch.tensor(np.stack([s["img"] for s in slices]))[:, None].float()
    y = torch.tensor(np.stack([s["mask"] for s in slices])).long()
    return x, y


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + 32].to(device)).argmax(1).cpu() for i in range(0, len(x), 32)])


def confusion(pred, y):
    return np.bincount((y.reshape(-1) * 4 + pred.reshape(-1)).numpy(), minlength=16).reshape(4, 4)


split = json.load(open(os.path.join(OUT, "split.json")))
data = {os.path.basename(p): load_patient(p)[0] for p in sorted(glob.glob(DATA + "/patient*"))}
tr = [s for n in split["train"] for s in data[n]]
va = [s for n in split["val"] for s in data[n]]
xt, yt = tensors(tr)
xv, yv = tensors(va)

model = UNet().to(device)
params = sum(p.numel() for p in model.parameters())
log(f"device {device}, {params} parameters, {len(tr)} train slices, {len(va)} val slices")
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

history, best, start = [], -1.0, time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    order, total = torch.randperm(len(xt)), 0.0
    for i in range(0, len(xt), BATCH):
        j = order[i:i + BATCH]
        opt.zero_grad()
        loss = loss_fn(model(xt[j].to(device)), yt[j].to(device))
        loss.backward()
        opt.step()
        total += loss.item() * len(j)
    sc = scores(confusion(predict(model, xv), yv))
    history.append(dict(epoch=epoch, train_loss=round(total / len(xt), 4), val_macro_f1=sc["macro_f1"],
                        seconds=round(time.time() - start, 1)))
    log(history[-1])
    if sc["macro_f1"] > best:
        best = sc["macro_f1"]
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "acdc_unet_pilot.pt"))

model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "acdc_unet_pilot.pt")))
t0 = time.time()
pred = predict(model, xv)
pred_s = time.time() - t0
cm = confusion(pred, yv)


def sub(cond):
    i = [k for k, s in enumerate(va) if cond(s)]
    return scores(confusion(pred[i], yv[i]))["macro_f1"]


third = lambda s: "top" if s["z"] < 1 / 3 else ("middle" if s["z"] < 2 / 3 else "bottom")
per_patient = {n: sub(lambda s, n=n: s["patient"] == n) for n in split["val"]}
report = dict(device=device, parameters=params, epochs=EPOCHS, batch=BATCH, learning_rate=LR,
              train_seconds=history[-1]["seconds"], predict_seconds_val=round(pred_s, 1),
              best_epoch=max(history, key=lambda h: h["val_macro_f1"])["epoch"],
              validation=scores(cm), confusion_matrix=cm.tolist(),
              by_group={g: sub(lambda s, g=g: s["group"] == g) for g in sorted({s["group"] for s in va})},
              by_moment={m: sub(lambda s, m=m: s["moment"] == m) for m in ("ED", "ES")},
              by_slice_position={p: sub(lambda s, p=p: third(s) == p) for p in ("top", "middle", "bottom")},
              per_patient=per_patient, history=history)
json.dump(report, open(os.path.join(OUT, "unet_pilot_results.json"), "w"), indent=1)

# Figure: the same three patients as the forest figure.
forest = json.load(open(os.path.join(OUT, "forest_results.json")))["per_patient"]
order = sorted(split["val"], key=forest.get)
pick = [order[0], order[len(order) // 2], order[-1]]
colours = ListedColormap(["#e4572e", "#f2c14e", "#2e86ab"])
fig, ax = plt.subplots(3, 3, figsize=(10, 10.5))
for r, n in enumerate(pick):
    idx = [i for i, s in enumerate(va) if s["patient"] == n and s["moment"] == "ED"]
    i = idx[len(idx) // 2]
    for c, (lab, title) in enumerate(((None, "MRI slice"), (va[i]["mask"], "Expert labels"), (pred[i].numpy(), "U-Net prediction"))):
        ax[r, c].imshow(va[i]["img"].T, cmap="gray")
        if lab is not None:
            ax[r, c].imshow(np.ma.masked_where(lab.T == 0, lab.T), cmap=colours, vmin=1, vmax=3, alpha=.6, interpolation="nearest")
        ax[r, c].set_title(title + (f"\n{n} ({va[i]['group']}), patient macro F1 {per_patient[n]:.2f}" if c == 0 else ""), fontsize=10)
        ax[r, c].axis("off")
fig.legend(handles=[Patch(color=colours(i), label=CLASSES[i + 1]) for i in range(3)], loc="lower center", ncol=3)
fig.savefig(os.path.join(OUT, "unet_pilot_examples.png"), dpi=120, bbox_inches="tight")
log("DONE", report["validation"])
