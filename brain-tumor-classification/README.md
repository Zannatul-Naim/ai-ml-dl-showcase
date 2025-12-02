# Brain Tumor Classification (MRI)

This repository contains a PyTorch-based pipeline for multi-class brain tumor classification using a Kaggle MRI dataset (glioma, meningioma, pituitary). The project covers preprocessing, model training, evaluation, and explainability (Grad-CAM, Grad-CAM++, Integrated Gradients).

---

## Contents
- `brain-tumor-classification-mri-dataset-kaggle.ipynb` — training, validation and evaluation pipeline (notebook).  
- `explainable-ai-visualization.ipynb` — implementations of Grad-CAM, Grad-CAM++, and Integrated Gradients for model interpretation.  
- `summary_visualization.png` — consolidated visualization of XAI results.

---

## Dataset
- **Source:** Kaggle Brain Tumor MRI Dataset (T1-weighted MRI images).  
- **Classes:** `glioma`, `meningioma`, `pituitary`.  
- **Preprocessing:** resize to 224×224, normalization, convert to tensors. Default train/validation/test split follows the dataset folder structure.

---

## Model & Training
- **Framework:** PyTorch  
- **Backbone:** Transfer-learning CNN (ResNet-style backbone adapted for 3 classes)  
- **Loss:** Cross-entropy  
- **Optimizer:** Adam  
- **Scheduler:** Step/ReduceLR (notebook contains exact configuration)  
- **Training regime:** 30 epochs (logs shown below)

---

## Training Log (validation accuracy)
    Epoch [1/30], Loss: 0.6553, Validation Accuracy: 88.63%
    Epoch [2/30], Loss: 0.5072, Validation Accuracy: 93.14%
    Epoch [3/30], Loss: 0.4664, Validation Accuracy: 94.05%
    Epoch [4/30], Loss: 0.4569, Validation Accuracy: 95.27%
    Epoch [5/30], Loss: 0.4361, Validation Accuracy: 93.97%
    Epoch [6/30], Loss: 0.4406, Validation Accuracy: 93.29%
    Epoch [7/30], Loss: 0.4245, Validation Accuracy: 94.97%
    Epoch [8/30], Loss: 0.4141, Validation Accuracy: 93.14%
    Epoch [9/30], Loss: 0.4145, Validation Accuracy: 97.48%
    Epoch [10/30], Loss: 0.3965, Validation Accuracy: 97.25%
    Epoch [11/30], Loss: 0.3959, Validation Accuracy: 96.11%
    Epoch [12/30], Loss: 0.3934, Validation Accuracy: 96.03%
    Epoch [13/30], Loss: 0.3833, Validation Accuracy: 97.10%
    Epoch [14/30], Loss: 0.3841, Validation Accuracy: 96.49%
    Epoch [15/30], Loss: 0.3723, Validation Accuracy: 98.70%
    Epoch [16/30], Loss: 0.3707, Validation Accuracy: 99.01%
    Epoch [17/30], Loss: 0.3683, Validation Accuracy: 99.01%
    Epoch [18/30], Loss: 0.3630, Validation Accuracy: 98.09%
    Epoch [19/30], Loss: 0.3646, Validation Accuracy: 99.24%
    Epoch [20/30], Loss: 0.3578, Validation Accuracy: 99.31%
    Epoch [21/30], Loss: 0.3550, Validation Accuracy: 99.62%
    Epoch [22/30], Loss: 0.3547, Validation Accuracy: 99.39%
    Epoch [23/30], Loss: 0.3542, Validation Accuracy: 99.47%
    Epoch [24/30], Loss: 0.3509, Validation Accuracy: 99.47%
    Epoch [25/30], Loss: 0.3523, Validation Accuracy: 99.54%
    Epoch [26/30], Loss: 0.3504, Validation Accuracy: 99.69%
    Epoch [27/30], Loss: 0.3514, Validation Accuracy: 99.77%
    Epoch [28/30], Loss: 0.3512, Validation Accuracy: 99.69%
    Epoch [29/30], Loss: 0.3505, Validation Accuracy: 99.85%
    Epoch [30/30], Loss: 0.3499, Validation Accuracy: 99.85%
## Training Finished. Best Validation Accuracy: 99.85%



---

## Results & Discussion
- **Best validation accuracy:** **99.85%** (achieved at epochs 29 and 30).  
- The training curve shows rapid improvement in early epochs and very high validation accuracy from epoch ~15 onward.  
- The XAI visualizations (Grad-CAM, Grad-CAM++, Integrated Gradients) consistently emphasize regions corresponding to tumor presence, supporting the claim that the network focuses on clinically relevant areas when making predictions (see `summary_visualization.png`).

**Notes:** the exceptionally high validation accuracy suggests strong model performance on the provided split. Evaluate model robustness further by:
- testing on an external hold-out dataset (different source / scanner / center),  
- using stratified cross-validation,  
- applying data augmentation and regularization (dropout, weight decay), and  
- reporting per-class precision, recall, F1-score and confusion matrix to ensure balanced performance.

---

## Explainability
Interpretability was performed using:
- Grad-CAM  
- Grad-CAM++  
- Integrated Gradients

![](./summary_visualization.png)

These methods are implemented in `explainable-ai-visualization.ipynb` and produce heatmaps overlaid on original MRIs. The combined visualization is included as `summary_visualization.png`.

---

## How to run
1. Install dependencies (see first cell in notebooks for exact versions).  
2. Open the notebooks in a Jupyter environment.  
3. Run the preprocessing and training cells in `brain-tumor-classification-mri-dataset-kaggle.ipynb`.  
4. Use `explainable-ai-visualization.ipynb` to generate interpretability visualizations for saved model checkpoints.

---