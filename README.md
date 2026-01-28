# Retinal Disease Classification using Transfer Learning

This project focuses on automated classification of retinal eye diseases using deep learning and transfer learning techniques. It demonstrates a complete end-to-end machine learning pipeline, from data preprocessing to model evaluation and interpretability.

---

## 🔍 Problem Statement
Early detection of retinal diseases is crucial to prevent vision loss. Manual diagnosis is time-consuming and requires expert ophthalmologists. This project aims to build an automated system to classify retinal images into multiple disease categories using convolutional neural networks.

---

## 🗄 Dataset & Preprocessing

- **Dataset:** Retinal images obtained from [Kaggle / publicly available dataset].  
- **Classes:** 5 retinal disease categories.  
- **Image Preprocessing:** All images were resized to 224x224 pixels and normalized.  
- **Data Augmentation:** Random rotations, horizontal and vertical flips, and zoom were applied to increase variability and improve generalization.  
- **Train/Validation/Test Split:** 70% training, 15% validation, and 15% test sets were used.

---

## 🧠 Methodology

Multiple state-of-the-art convolutional neural network architectures were implemented and compared using transfer learning:

- VGG16  
- VGG19  
- ResNet50  
- EfficientNetB3  
- DenseNet121  

All models were trained using the same preprocessing pipeline, training strategy, and evaluation metrics to ensure fair comparison.

---

## 📊 Model Comparison & Results

| Model Name                | Accuracy | Precision | Recall | F1-score |
|---------------------------|----------|-----------|--------|----------|
| VGG16                     | 89%      | 90%       | 89%    | 89%      |
| VGG19                     | 92%      | 92%       | 91%    | 91%      |
| ResNet50                  | 92%      | 91%       | 92%    | 92%      |
| EfficientNetB3            | 93%      | 93%       | 93%    | 92%      |
| DenseNet121 (Frozen)      | 90%      | 91%       | 90%    | 90%      |
| **Fine-tuned DenseNet121** | **95%** | **96%** | **95%** | **95%** |

While EfficientNetB3 performed best among frozen models, **fine-tuning DenseNet121** led to the highest overall performance, improving accuracy from 90% to 95%.

---

## 🔧 Fine-Tuning Strategy

DenseNet121 was initially evaluated as a frozen pre-trained model.  
To improve performance, deeper layers were unfrozen and retrained using a lower learning rate, allowing the model to learn domain-specific retinal features.  
This fine-tuning strategy significantly improved accuracy, recall, precision, and F1-score.

---

## 🌍 External Validation & Grad-CAM

The final fine-tuned DenseNet121 model was evaluated on an **external validation dataset** to assess robustness and generalization.  
Gradient-weighted Class Activation Mapping (Grad-CAM) was applied to visualize the regions of retinal images most influential for predictions.  
This provides interpretability and highlights clinically relevant areas contributing to the model's decisions.

---

## 🧪 Evaluation

Model performance was analyzed using:

- Classification reports (precision, recall, F1-score)  
- Confusion matrix visualization  
- External validation results  

Evaluation utilities are implemented in `src/evaluate.py` for reproducibility.

---


---

## 🚀 How to Run

Follow these steps to set up and run the project:

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/retinal_disease_classification.git
cd retinal_disease_classification


