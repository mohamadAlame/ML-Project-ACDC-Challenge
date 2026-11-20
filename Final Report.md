# ACDC Cardiac MRI Segmentation and Diagnosis

## Abstract

This project studies whether machine learning can automate two connected tasks in cardiac MRI: segmenting the right ventricle, myocardium, and left ventricle, and using measurements derived from those segmentations to classify patients into five clinical groups. The project originally selected VinDr-CXR, but access to that dataset was not obtained in time. We therefore switched to the instructor-approved ACDC Cardiac MRI dataset and documented the consequences of that change.

We compared a classical random-forest baseline with a small U-Net convolutional neural network. The forest used handcrafted pixel features and achieved validation macro-F1 of 0.59 and test macro-F1 of 0.54. Its feature importance and failure cases showed that it relied heavily on the expected position of the heart and did not represent cardiac shape well. The U-Net used the same patient-level split and achieved 0.92 macro-F1 on the untouched test set. Its predicted outlines produced diagnosis accuracy of 0.88, compared with 0.92 using expert outlines and 0.40 using forest outlines. The results support the use of a spatial CNN for this task, while also showing that the small dataset and single U-Net validation split limit how broadly the result should be generalized.

## 1. Introduction

The project question is whether a model can produce useful cardiac measurements from MRI without relying on hand-drawn expert outlines. The question has two stages. First, the model must identify the cardiac structures at pixel level. Second, measurements such as ventricular volumes and ejection fraction can be calculated from those outlines and used for patient-group diagnosis.

The original proposal selected VinDr-CXR, an instructor-proposed chest X-ray project. Access required training and credentials, and the access process did not resolve in time. The group therefore moved to ACDC after the instructor announced it as an allowed alternative for students having difficulty accessing VinDr-CXR. This was not only a change of dataset: it changed the image task from chest-X-ray abnormality classification to cardiac segmentation followed by diagnosis. We kept the same project logic: define the problem, establish a conventional baseline, investigate a more appropriate image model, analyze failure modes, and measure whether the additional complexity improves the useful end-to-end result.

The main comparison is between a random forest and a small U-Net. The forest is a deliberately simple classical baseline. The U-Net is appropriate because CNNs exploit locality, shared filters, and spatial structure, all of which matter when the output is a pixel-level outline.

## 2. Dataset and preprocessing

ACDC contains 150 patients: 100 training patients and 50 test patients, divided into five groups: normal, previous myocardial infarction, dilated cardiomyopathy, hypertrophic cardiomyopathy, and abnormal right ventricle. The scans are stored in NIfTI format and include expert outlines for the right ventricle, myocardium, and left ventricle at end-diastolic and end-systolic moments.

The preprocessing pipeline was designed around the physical and statistical properties of MRI:

- Intensities were clipped at the extreme 1% of each scan and scaled to 0–1 because MRI brightness does not have one fixed scale across patients.
- Scans were resampled to 1.5 mm per pixel so that anatomical size was more comparable across patients.
- Slices were cropped or padded to 192 × 192, while checking that outlined pixels were retained.
- The data was split by patient: 80 training patients and 20 validation patients, with four validation patients per group. Slices from one patient were never divided between training and validation.
- The 50 official test patients remained untouched until the final evaluation.

For the forest, each pixel became one row containing 11 features: intensity, local brightness features at several scales, edge information, relative brightness, spatial coordinates, and slice position. Because most pixels are background, training used sampled pixels rather than every pixel. The final sampling configuration used up to 300 foreground pixels and 1,200 background pixels per slice.

For diagnosis, measurements were calculated from the expert outlines. These included ventricular volumes, ejection fractions, muscle mass, body-size-normalized values, the right-to-left volume ratio, and maximum wall thickness. The measurements were standardized before logistic regression.

## 3. Models and experimental design

### 3.1 Classical segmentation baseline

The first segmentation models were logistic regression, a decision tree, a random forest, and gradient boosting, all using the same pixel features. The random forest and gradient boosting results were similar, but the forest was substantially faster at prediction, so it became the classical baseline.

The forest was then tuned on the validation patients. The experiments varied foreground/background sampling, class weighting, depth, leaf size, feature subsampling, and number of trees. Class weighting made the result worse, while providing more background examples improved macro-F1. A five-fold patient-level check gave 0.584 ± 0.020 macro-F1, supporting the conclusion that the baseline was weak rather than the product of one unusually bad split.

### 3.2 U-Net segmentation model

The sophisticated model was a small U-Net trained from scratch. It uses convolutional layers, ReLU activations, downsampling, and skip connections so that it can combine local features with larger-scale shape information. It was trained on the same patient-level training split and evaluated on the same validation patients.

The variants tested were:

| Variant | Validation macro-F1 | Parameters | Training time |
|---|---:|---:|---:|
| Pilot: cross-entropy, no augmentation | 0.902 | 483,204 | 10.3 min |
| Augmentation | 0.909 | 483,204 | 14.6 min |
| Augmentation + Dice loss | 0.914 | 483,204 | 14.7 min |
| Wider network, augmentation + Dice | 0.915 | 1,927,940 | 31.3 min |
| Augmentation + Dice, 80 epochs | **0.920** | 483,204 | 30.3 min |

The final choice was the 80-epoch model because it achieved the best validation score while remaining small. The wider model gained only 0.001 while using approximately four times as many parameters and twice the training time.

### 3.3 Diagnosis models

For diagnosis, logistic regression, a decision tree, a random forest, and a small neural network were evaluated using repeated stratified cross-validation on the 100 labelled training patients. Logistic regression was selected as the classical diagnosis model because it performed as well as the more complex alternatives while remaining easier to interpret.

Diagnosis was then evaluated using measurements from expert outlines, U-Net outlines, and forest outlines. This tests the complete pipeline rather than treating segmentation and diagnosis as unrelated tasks.

## 4. Results and analysis

### 4.1 Segmentation comparison

| Model | Validation macro-F1 | Test macro-F1 | Test pixel accuracy |
|---|---:|---:|---:|
| Random forest | 0.590 | 0.539 | 0.950 |
| U-Net | 0.920 | **0.920** | 0.993 |

The forest’s per-structure test F1 scores were 0.51 for the right ventricle, 0.49 for the myocardium, and 0.61 for the left ventricle. The U-Net scores were 0.92, 0.89, and 0.95 respectively.

Accuracy alone would have been misleading because background dominates the image. Predicting background everywhere gives approximately 94.8% test pixel accuracy, while finding no cardiac structures. Macro-F1 therefore better reflects the actual segmentation task.

### 4.2 Baseline failure analysis

Position features accounted for approximately 62% of the forest’s feature importance. When those features were removed, validation macro-F1 fell from 0.59 to 0.15. The worst validation patient had macro-F1 of 0.35 and an off-centre heart; the forest marked the expected central region rather than learning the cardiac shape.

This failure explains why the U-Net was a justified next step. The U-Net receives the image as an image and can use neighbouring pixels, local boundaries, and the global arrangement of the structures. The forest receives isolated feature rows and cannot naturally represent the shape of the complete heart.

### 4.3 U-Net failure analysis and mitigations

We also looked for where the U-Net stops working. Across the five patient groups its test macro-F1 stayed between 0.91 and 0.92, so no diagnosis group was a clear weak spot. The failures were concentrated elsewhere:

| Test subset | U-Net macro-F1 |
|---|---:|
| Middle third of the slice stack | 0.94 |
| First third of the slice stack | 0.92 |
| Last third of the slice stack | **0.89** |
| Full heart (end-diastole) | 0.93 |
| Squeezed heart (end-systole) | **0.91** |

The weakest region is the last third of each stack, where the structures become small and their boundaries blurrier, and the squeezed scans, where the chambers are smaller. In both cases a few wrong pixels are a larger share of a small structure. The worst test patient scored 0.86.

Two of our changes were mitigations aimed at observed failures, and both were evaluated on the validation patients. Augmentation (random small rotations, shifts and zooms) targeted the reliance on position that broke the forest, and raised macro-F1 from 0.902 to 0.909. Dice loss targeted the small structures, which are about 1% of the pixels each, and raised it to 0.914. Together they also made training steadier: the pilot dipped to 0.85 during training, while the worst of the last five epochs with both changes was 0.906. Longer training then reached 0.920.

### 4.4 End-to-end diagnosis

| Outline source | Diagnosis accuracy |
|---|---:|
| Expert outlines | 0.92 |
| U-Net outlines | **0.88** |
| Forest outlines | 0.40 |

The result shows why segmentation quality matters. The U-Net’s outlines preserve most of the useful diagnostic signal. The forest’s poor outlines create inaccurate volumes and ejection fractions, and the downstream diagnosis collapses.

The mean absolute ejection-fraction error from U-Net outlines was 2.4 percentage points for the left ventricle and 4.7 points for the right ventricle. With forest outlines, the errors were 26.0 and 35.4 points.

### 4.5 Confidence and complexity

The U-Net had a test calibration error of 0.054, compared with 0.22 for the forest. Its average confidence was also lower on incorrect predictions than on correct ones, which gives some signal for identifying uncertain cases.

| Model | Parameters | Training time | Prediction time per slice | Model file | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Random forest (200 trees) | — | 3 min per fit; about 1 h with tuning and 5-fold check | 0.095 s | 272 MB | 0.54 (test) |
| U-Net pilot | 483,204 | 10 min | — | 2 MB | 0.902 (validation) |
| Wider U-Net | 1,927,940 | 31 min | — | 8 MB | 0.915 (validation) |
| **Final U-Net** | **483,204** | **30 min** | **0.004 s** | **2 MB** | **0.920 (test)** |

The forest is the classical model, but it is not the cheaper one in every sense. Its saved model is more than 100 times larger than the U-Net's, and it predicts about 20 times slower, because every pixel needs its own feature row and a pass through 200 trees. The U-Net costs more to train, but only once. Making the U-Net wider did not justify its cost: four times the parameters and twice the training time for a validation gain of 0.001. The useful improvement came from using a spatial model, augmentation, Dice loss, and longer training rather than from a bigger network.

## 5. Limitations

The dataset contains only 100 labelled training patients, so patient-level variation matters. The U-Net experiments used one fixed patient-level train/validation split. This is a valid holdout design, but a different set of 20 validation patients could produce a different score.

The U-Net was not evaluated with a complete five-fold cross-validation study. Such a study would require retraining the network across multiple patient folds and would provide a better estimate of split-to-split stability. The random forest did receive a five-fold patient-level check, but the U-Net result should be described as strong evidence on this held-out test set rather than as a fully general estimate of performance.

The best U-Net epoch was selected using validation macro-F1. This is standard early stopping, but selecting the best score from many epochs can make the validation result slightly optimistic. The final test patients were not used for epoch selection and were evaluated only after the model choice was made.

The diagnosis experiment using expert outlines is a best-case reference because the experts provide the measurements directly. The more meaningful end-to-end result is the lower 0.88 accuracy obtained from U-Net-generated outlines.

## 6. Conclusion

The project began with an access failure for the proposed VinDr-CXR dataset and was re-scoped to ACDC Cardiac MRI using an instructor-approved alternative. The resulting process followed a complete machine-learning workflow: data audit, preprocessing, patient-level evaluation design, classical baseline, failure analysis, spatial CNN, ablation experiments, untouched test evaluation, and downstream analysis.

The random forest was useful as a baseline because it revealed a concrete failure mode: it relied on where the heart usually appeared and did not represent cardiac shape. The U-Net addressed that limitation and improved test macro-F1 from 0.54 to 0.92. More importantly, its predicted outlines preserved most of the diagnostic performance available from expert outlines, while the forest’s outlines did not.

The conclusion is therefore not that a more complex model is always better. It is that, for this image-segmentation problem, the additional spatial structure of a small U-Net was justified by a large improvement in segmentation, measurement accuracy, and end-to-end diagnosis. The remaining uncertainty comes mainly from the small number of labelled patients and the single U-Net validation split.

## Code and model

Code, results and the final trained model: https://github.com/mohamadAlame/ML-Project-ACDC-Challenge

## References

1. Bernard, O. et al. “Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?” *IEEE Transactions on Medical Imaging*, 2018.
2. Rida Assaf. *Data Preprocessing and Evaluation Metrics*, *Convolutional Neural Networks Parts 1–2*, course slides, CMPS 261, AUB, 2026.
3. Rida Assaf and Aya Kayal. *CMPS 261 Course Project*, AUB, Fall 2026.
