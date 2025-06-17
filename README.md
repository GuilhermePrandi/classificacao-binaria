# 🫀 Binary Classification: Ischemic Heart Disease (IHD)

This project implements two machine learning models, **Naive Bayes** and **MLP** to perform **binary classification** of the presence or absence of **Ischemic Heart Disease (IHD)** based on clinical and lifestyle variables.

---

## 📂 Dataset

- **Source:** [Kaggle - Quantum Enhanced Ischemic Heart Disease Dataset](https://www.kaggle.com/datasets/ziya07/quantum-enhanced-ischemic-heart-disease-dataset)
- **Target:** `IHD` → 0 (absence) or 1 (presence) of ischemic heart disease
- **Features used:**
  - Age
  - Gender
  - Systolic Blood Pressure
  - Cholesterol
  - BMI
  - Smoking Status
  - Physical Activity
  - Diabetes
  - Hypertension

---

## 🧠 Models Used

### ✅ Naive Bayes
- Probabilistic algorithm (`GaussianNB`)
- Assumes independence between features

### ✅ MLP (Multi-Layer Perceptron)
- Architecture:
  - 2 hidden layers: 16 and 8 neurons
  - Activation function: ReLU
  - Optimizer: Adam
  - Max iterations: 10,000

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/binary-classification-ihd.git
cd binary-classification-ihd
python classificacao_ihd.py
