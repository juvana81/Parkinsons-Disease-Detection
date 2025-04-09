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
- [🚀 Getting Started](#-getting-started)
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

- Count Plot: Healthy vs Parkinson's cases
![image](https://github.com/user-attachments/assets/763b36d7-d9e3-4a60-886e-798625379c84)


- Correlation Heatmap:  Feature correlation.
![image](https://github.com/user-attachments/assets/a8d96925-9da8-4823-b8b3-8a23c12e7ce5)


- Data Distribution and Histograms
![image](https://github.com/user-attachments/assets/e37910ce-6785-4403-ab2d-d63edfcf8643)


- Boxplots by status: Comparing Jitter values across `status`.
![image](https://github.com/user-attachments/assets/d305b903-8d49-4aec-9ac4-080650c19ff7)


- Pairplots for feature clusters
![image](https://github.com/user-attachments/assets/3b421db3-049a-4625-bb58-83aa4a080cf8)




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

#### 📊 Example Results:
| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 88.5%    |
| Random Forest       | 91.3%    |
| SVM (RBF)           | 89.7%    |
| KNN                 | 86.1%    |
| Gradient Boosting   | **92.3%** ✅ |

---

## 🌀 Part 3: CNN for Spiral Image Detection

We built a custom Convolutional Neural Network (CNN) using PyTorch to classify hand-drawn spiral images.

### 🧠 CNN Architecture:
- **3x Conv2D layers** with ReLU + MaxPool
- **BatchNorm2D** after each conv block
- **Flatten → Fully Connected (FC) → Dropout → Sigmoid output**

### 🔄 Steps:
1. Prepare spiral image dataset.
2. Load datasets with `ImageFolder` & `DataLoader`.
3. Train CNN in `train.py`.
4. Evaluate performance in `evaluate.py`.

### 🧪 Metrics:
- Binary Accuracy
- F1-Score
- Precision & Recall

---

## 🛠 Tech Stack

- **Programming Language:** Python 3.9+
- **Libraries for ML & Visualization:** `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`
- **Deep Learning & Image Processing:** `torch`, `torchvision`, `PIL`
- **IDE:** Jupyter Notebook, Google Colab, VS Code

---

## 🚀 Getting Started

### 🔧 Clone the Repository:
```bash
git clone https://github.com/your-username/parkinsons-full-pipeline.git
cd parkinsons-full-pipeline
```

### 📦 Install Dependencies:
```bash
pip install -r requirements.txt
```

### 💻 Run the Project:
```bash
# For voice data analysis and ML
jupyter notebook voice_analysis.ipynb

# For CNN training on spiral images
python train.py
python evaluate.py
```

---

## 📈 Results & Comparison

### Voice-Based ML Models:
- **Best accuracy:** 92.3% (Gradient Boosting)
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
