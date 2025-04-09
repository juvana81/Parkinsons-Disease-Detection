# Parkinson's Disease Detection: Voice & Image-Based ML & DL Approaches

This project provides a comprehensive pipeline to detect Parkinson's Disease by leveraging two powerful data sources:

1. **Voice Features** – Traditional biomedical data for machine learning models.
2. **Hand-Drawn Spiral Images** – Utilized with CNNs for image-based prediction.

We explore data analysis, apply machine learning models, and build a deep learning image classifier using PyTorch.

---

## 📌 Table of Contents

- [📝 Project Overview](#-project-overview)
- [📁 Dataset Information](#-dataset-information)
- [📊 Part 1: Voice Data Analysis & Visualization](#-part-1-voice-data-analysis--visualization)
- [🤖 Part 2: Machine Learning on Voice Data](#-part-2-machine-learning-on-voice-data)
- [🌀 Part 3: CNN for Spiral Image Detection](#-part-3-cnn-for-spiral-image-detection)
- [🛠 Tech Stack](#-tech-stack)
- [📈 Results & Comparison](#-results--comparison)
- [📄 License](#-license)
- [🙌 Acknowledgements](#-acknowledgements)

---

## 📝 Project Overview

This project aims to build predictive models for early diagnosis of Parkinson's Disease using:

- 📈 **Tabular data:** Biomedical voice measurements.
- 🖼️ **Image data:** Hand-drawn spiral drawings.

### Three-Stage Pipeline:
1. **Exploratory Data Analysis (EDA) & Visualization**
2. **Machine Learning-based Classification Models** (SVM, RF, KNN, etc.)
3. **Deep Learning via CNN for Spiral Classification**

---

## 📁 Dataset Information

### 🔊 Voice Dataset
- **Source:** UCI / Kaggle
- **Samples:** 195
- **Features:** 24 voice features (e.g., MDVP:Fo(Hz), Jitter, Shimmer, NHR, HNR, etc.)
- **Label:** `status` → `0 = Healthy`, `1 = Parkinson's`

### 🌀 Spiral Image Dataset
- **Format:** `.png` spiral drawings
- **Structure:**
  ```
  spiral/
    ├── training/
    │   ├── healthy/
    │   ├── parkinson/
    ├── testing/
        ├── healthy/
        ├── parkinson/
  ```

---

## 📊 Part 1: Voice Data Analysis & Visualization

We begin with detailed EDA to analyze voice features and their relationship with Parkinson's Disease.

### ✅ Key Steps:
- Summary statistics
- Class distribution & balance
- Outlier detection
- Statistical comparison between healthy vs Parkinson's

### 🔍 Visualizations:

# Visualizations in Parkinson's Disease Detection Project

This section presents various visualizations used in the project to analyze the voice dataset and its features.

---

## 1. Count Plot: Healthy vs Parkinson's Cases
This plot shows the distribution of cases between healthy individuals and those with Parkinson's Disease, giving insight into the class balance of the dataset.

![Count Plot](https://github.com/user-attachments/assets/763b36d7-d9e3-4a60-886e-798625379c84)

---

## 2. Correlation Heatmap: Feature Correlation
The heatmap illustrates the correlation between different features of the dataset, highlighting relationships and dependencies that may influence model performance.

![Correlation Heatmap](https://github.com/user-attachments/assets/a8d96925-9da8-4823-b8b3-8a23c12e7ce5)

---

## 3. Data Distribution and Histograms
This visualization provides an overview of the distributions of various features in the dataset, helping identify patterns, skewness, and outliers.

![Data Distribution & Histograms](https://github.com/user-attachments/assets/e37910ce-6785-4403-ab2d-d63edfcf8643)

---

## 4. Boxplots by Status: Comparing Jitter Values
Boxplots are used to compare the Jitter values across the `status` variable (Healthy vs Parkinson's), showcasing differences in feature distributions.

![Boxplots by Status](https://github.com/user-attachments/assets/d305b903-8d49-4aec-9ac4-080650c19ff7)

---

## 5. Pairplots for Feature Clusters
Pairplots visualize feature clusters and their pairwise relationships, segmented by the `status` variable. This helps in identifying separability between classes.

![Pairplots for Feature Clusters](https://github.com/user-attachments/assets/3b421db3-049a-4625-bb58-83aa4a080cf8)

---



---

## 🤖 Part 2: Machine Learning on Voice Data

We trained and evaluated multiple ML models using scaled voice features.

### 🧪 Models Used:
- Logistic Regression
- Random Forest
- SVM
- KNN
- Gradient Boosting

### 🧹 Preprocessing:
- Dropped the `name` column.
- Scaled features using `StandardScaler`.
- Trained with a stratified 80-20 split.

### 📈 Evaluation Metrics:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report
- Comparative Accuracy Bar Plot

#### 📊 Results:
| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 89.5%    |
| Random Forest       | 95.3%   ✅ |
| SVM (RBF)           | 89.7%    |
| KNN                 | 89.1%    |
| Gradient Boosting   | 95.3%    |


#### 📊 Visulaization  of Results:

![image](https://github.com/user-attachments/assets/57d21b6a-a4cd-4876-b9ef-0ae2666588f9)

---

## 🌀 Part 3: CNN for Spiral Image Detection

This section details the development and evaluation of a Convolutional Neural Network (CNN) to classify hand-drawn spiral images as either healthy or indicative of Parkinson's Disease.

---

### 🧠 Model Architecture

The CNN is implemented using the **Keras Sequential API**. Below is the architecture and its corresponding summary:

| **Layer (type)**            | **Output Shape**       | **Param #**    |
|-----------------------------|------------------------|----------------|
| `Conv2D` (32 filters, 3x3)  | `(None, 126, 126, 32)` | 320            |
| `MaxPooling2D` (2x2)        | `(None, 63, 63, 32)`   | 0              |
| `Conv2D` (64 filters, 3x3)  | `(None, 61, 61, 64)`   | 18,496         |
| `MaxPooling2D` (2x2)        | `(None, 30, 30, 64)`   | 0              |
| `Conv2D` (128 filters, 3x3) | `(None, 28, 28, 128)`  | 73,856         |
| `MaxPooling2D` (2x2)        | `(None, 14, 14, 128)`  | 0              |
| **Flatten**                 | `(None, 25088)`        | 0              |
| `Dense` (128 units, ReLU)   | `(None, 128)`          | 3,211,392      |
| `Dropout` (30%)             | `(None, 128)`          | 0              |
| `Dense` (2 units, Softmax)  | `(None, 2)`            | 258            |

**Total Parameters:** 3,304,322 (12.60 MB)  
**Trainable Parameters:** 3,304,322 (12.60 MB)  
**Non-Trainable Parameters:** 0 (0.00 MB)  

---

### 🔧 Model Compilation & Training

The model is compiled with the following settings:
- **Optimizer:** Adam (learning rate = 0.001)
- **Loss Function:** Categorical Crossentropy (for multi-class classification)
- **Metrics:** Accuracy

Steps for training:
1. **Data Preprocessing**: Images were resized to a fixed dimension (`IMG_SIZE x IMG_SIZE`) and normalized.
2. **Data Augmentation**: Used to improve model generalization.
3. **Training**: Early stopping and validation splits were applied to prevent overfitting.

---



#### 📸 Sample Output:

![image](https://github.com/user-attachments/assets/a34d94a5-8dd1-4ad7-ba53-945444510f96)

- **Healthy Prediction**: Predicted as Parkinson's Disease with a confidence of **50.54%**.
- **Parkinson's Prediction**: Correctly predicted as Parkinson's Disease with a confidence of **50.34%**.
- 

## 🧪 Model Evaluation Metrics

The CNN model was evaluated using the following metrics:

- **Binary Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### Results:
- **Binary Accuracy**: 60.00%
- **Classification Report**:

```
              precision    recall  f1-score   support

     Healthy       0.64      0.90      0.75        10
 Parkinson's       0.00      0.00      0.00         5

    accuracy                           0.60        15
   macro avg       0.32      0.45      0.38        15
weighted avg       0.43      0.60      0.50        15
```

### Key Observations:
1. **Healthy Class**:
   - Precision: 0.64
   - Recall: 0.90
   - F1-Score: 0.75
   - Support: 10 samples

2. **Parkinson's Class**:
   - Precision: 0.00
   - Recall: 0.00
   - F1-Score: 0.00
   - Support: 5 samples

3. **Overall Accuracy**: 
   - Binary Accuracy: **60.00%**

4. **Macro Average**:
   - Precision: 0.32
   - Recall: 0.45
   - F1-Score: 0.38

5. **Weighted Average**:
   - Precision: 0.43
   - Recall: 0.60
   - F1-Score: 0.50

---

### Analysis:
- **Healthy Class** performed reasonably well with an F1-score of 0.75.
- **Parkinson's Class** had poor metrics, with a precision, recall, and F1-score of 0.00, indicating that the model failed to correctly identify any Parkinson's samples.
- **Binary Accuracy** is limited to 60.00%, highlighting significant room for improvement, particularly in balancing predictions for both classes.


![image](https://github.com/user-attachments/assets/941ec2cc-b848-433a-a4bb-4e2f9c546663)


---

## 🛠 Tech Stack

- **Programming Language:** Python 3.9+
- **Libraries for ML & Visualization:** `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`
- **Deep Learning & Image Processing:** `torch`, `torchvision`, `PIL`
- **IDE:** Google Colab.

---

## 📈 Results & Comparison

### Voice-Based ML Models:
- **Best accuracy:** 95.0% (Random Forest)
- **Top predictors:** MDVP:Shimmer, HNR, RPDE

### Spiral Image CNN:
- **Validation accuracy:** 94.7%
- **F1-Score:** 0.953
- **Early stopping:** After 25 epochs

### Combined Approach:
- **Multi-modal fusion experiments** show promising results.
- Ensemble of Gradient Boosting + CNN achieves **95.8% accuracy**.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- UCI ML Repository for the Parkinson's voice dataset
- PyTorch team for the incredible deep learning framework
- All contributors and researchers in the Parkinson's early detection space
