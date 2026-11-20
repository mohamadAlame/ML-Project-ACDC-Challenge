# ACDC Cardiac MRI: Segmentation and Diagnosis

CMPS 261 (Machine Learning), Fall 2026, American University of Beirut. Group 26: Mohamad Alame, Sarra Fassakh.

We outline the heart in cardiac MRI slices (right ventricle, myocardium, left ventricle), then use the outlines to measure the heart and predict the patient's diagnostic group.

## Pipeline

1. **Segmentation:** MRI slice → label for every pixel.
   - Classical baseline: random forest on hand-made pixel features.
   - Final model: a small U-Net trained from scratch.
2. **Measurements:** outlines → volumes, ejection fractions, muscle mass, wall thickness.
3. **Diagnosis:** measurements → one of five groups (normal, previous myocardial infarction, dilated cardiomyopathy, hypertrophic cardiomyopathy, abnormal right ventricle), using logistic regression.

## Results on the 50 test patients

| | Random forest | U-Net |
|---|---:|---:|
| Macro F1, 3 heart structures | 0.54 | **0.92** |
| Diagnosis accuracy from its outlines | 0.40 | **0.88** |
| Prediction time per slice | 0.095 s | 0.004 s |

Diagnosis from the experts' outlines reaches 0.92.

All model choices were made on 20 validation patients. The test set was used once. Full numbers are in `results/experiments.json`.

## Files

| File | What it does |
|---|---|
| `Group 26 Final Project Notebook.ipynb` | All code in one notebook, with the result tables |
| `acdc_forest_baseline.py` | Preprocessing, random forest tuning, 5-fold check |
| `acdc_diagnosis_baseline.py` | Heart measurements and diagnosis models |
| `acdc_unet_pilot.py` | First U-Net run |
| `acdc_experiments.py` | U-Net variants, diagnosis tuning, final test |
| `results/` | Scores, tuning tables, split, logs and figures |
| `cmps261-work/models/acdc_unet_E.pt` | **Final trained U-Net** (PyTorch weights) |
| `cmps261-work/models/acdc_diagnosis_final.joblib` | Final diagnosis model (scikit-learn) |
| `Final Report.md`, `Group 26 Final Report.docx` | Report |

The random forest model (272 MB) is over GitHub's file limit, so it isn't included. `acdc_forest_baseline.py` rebuilds it.

## How to run

1. Install Python 3.11 with `numpy scipy scikit-learn torch nibabel matplotlib pandas joblib`.
2. Download ACDC from https://humanheart-project.creatis.insa-lyon.fr/database/ and extract it so the patient folders sit at `cmps261-work/data/acdc/extracted/database/training/` and `.../testing/`. You can also set `ACDC_WORK` to another folder.
3. Run these in order:

```bash
python acdc_forest_baseline.py
python acdc_diagnosis_baseline.py
python acdc_unet_pilot.py
python acdc_experiments.py
```

Training uses random shuffling and augmentation, so a rerun gives slightly different numbers. The full set took about 2.5 hours on a laptop. To load the final U-Net, use the `UNet` class in `acdc_experiments.py` with `UNet(16)` and `load_state_dict`.

## Data and citation

ACDC is licensed CC BY-NC-SA 4.0. Please cite:

O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al. "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?" IEEE Transactions on Medical Imaging, 37(11), 2514–2525, 2018. doi:10.1109/TMI.2018.2837502
