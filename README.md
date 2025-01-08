# Confidence-based early stopping in motor imagery BCI
This repository contains the implementation and results of my bachelor project: [Confidence-Based Early Classification for Motor Imagery BCI](https://fse.studenttheses.ub.rug.nl/33772/1/bAI_2024_HassanM.pdf), supervised by I.P. de Jong, MSc. The project investigates a new confidence-based early classification approach for Motor Imagery (MI) Brain-Computer Interfaces (BCIs), focusing on optimizing earliness and accuracy.

# Overview
BCI technology enables direct communication between the brain and external devices. This project introduces a confidence-based dynamic classification model for MI  BCI systems. The dynamic model utilizes confidence thresholds and a stopping criterion to make early classification decisions, aiming to improve performance and timeliness.

# Dataset
 - [BCI Competition IV 2a](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html)
   - 9 subjects
   - 4 classes: Left hand, right hand, tongue, legs
   - Trial length: 4 sec
   - Total trials: 62208 
   - Sample frequency: 250 Hz

# Methods
## Preprocessing
- Bandpass Filtering: Isolate EEG frequency bands relevant for MI (mu: 7–13 Hz, beta: 13–30 Hz).
- Window Segmentation: Both sliding and expanding window approaches are implemented and assessed.

## Feature Extraction
- Common Spatial Pattern (CSP): Extract discriminative features for classification by learning the optimal spatial filters.

## Dynamic early classification
The model uses Linear Discriminant Analysis (LDA) or Support Vector Machines (SVM) as classifiers. Early classification happens as follows (for the dynamic model):
 1. The model calculates the prediction confidence at each time window by estimating the predictive entropy, which quantifies the uncertainty of the classifier's prediction.
 2. The model uses a tuned **Patience** parameter, which defines how many consecutive time windows need to exceed the confidence threshold before the model makes a classification.
 3. Once the model observes a **Patience** number of consecutive windows with a predictive entropy above the threshold, the highest probability class is predicted. This allows the model to predict the motor intention of the subject as soon as the confidence is sufficiently high, without needing the full trial data.

# Evaluation Metrics
- Averaged accuracy: Measures the mean ratio of correct predictions over the total number of predictions of classes.
- Cohen's Kappa:  Measures the agreement between the true label and predicted labels of the model.
- Information Transfer Rate (ITR): Measures the overall performance of BCIs by considering both accuracy and prediction time.

# Results
## Sliding window
| **Metric**        | LDA Dynamic | LDA Static | SVM Dynamic | SVM Static |
|--------------------|-------------|------------|-------------|------------|
| Accuracy (%)       | 56.4        | 54.7       | 52.9        | 51.5       |
| Kappa              | 0.42        | 0.40       | 0.37        | 0.35       |
| ITR (bits/min)     | 9.27        | 8.48       | 7.56        | 7.11       |

## Expanding window
| **Metric**        | LDA Dynamic | LDA Static | SVM Dynamic | SVM Static |
|--------------------|-------------|------------|-------------|------------|
| Accuracy (%)       | 58.0        | 58.0       | 57.5        | 57.5       |
| Kappa              | 0.44        | 0.44       | 0.43        | 0.43       |
| ITR (bits/min)     | 7.41        | 7.41       | 7.09        | 7.10       |

## Future Work
- Extend to additional MI datasets for generalization.
- Explore real-world asynchronous BCI scenarios.
- Introduce subject-specific tuning.
- Incorporate more robust uncertainty quantification methods for epistemic uncertainty.

## Usage instructions

Clone the repository

```
git clone https://github.com/MaxXassan/Confidence-based-Early-Classification-for-Motor-Imagery-BCI.
```

# Project Structure

```
Confidence-based-Early-Classification-for-Motor-Imagery-BCI
    ├───README.md
    │
    └───Early_predict_UQ
        ├───requirements.txt - Python package dependencies
        │
        ├───data
        │   └───make_dataset.py - Preprocesses and epochs the dataset
        │
        └───models
            ├───LDA_models
            │   ├───DynVsStat_LDA_expanding.py - Dynamic model vs static model comparison (expanding windows LDA)
            │   ├───DynVsStat_LDA_sliding.py - Dynamic model vs static model comparison (sliding windows LDA)
            │   ├───Tune_LDA_expanding.py - Hyperparameter tuning (expanding windows LDA)
            │   └───Tune_LDA_sliding.py - Hyperparameter tuning (sliding windows LDA)
            │
            ├───SVM_models
            │   ├───DynVsStat_SVM_expanding.py - Dynamic model vs static model comparison (expanding windows SVM)
            │   ├───DynVsStat_SVM_sliding.py - Dynamic model vs static model comparison (sliding windows SVM)
            │   ├───Tune_SVM_expanding.py - Hyperparameter tuning (expanding windows SVM)
            │   └───Tune_SVM_sliding.py - Hyperparameter tuning (sliding windows SVM)
            │
            └───Main.py - Script for possibly running all models; recommend running each model separately due to long runtime.
    │
    └───reports
        ├───figures
        │   ├───cumulative
        │   │   ├───LDA/dynamicVSstatic/ - Contains LDA model plots and metrics
        │   │   └───SVM/dynamicVSstatic/ - Contains SVM model plots and metrics
        │
        └───thesis
            └───Bachelor_s_project___Motor_Imagery_BCI.pdf - Thesis document

```