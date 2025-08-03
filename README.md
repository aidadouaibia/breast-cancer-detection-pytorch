# Breast Cancer Detection using Neural Networks (PyTorch)

This project applies a feedforward neural network (ANN) built with PyTorch to classify breast cancer tumors as **benign** or **malignant** based on 30 medical features (e.g., radius, concavity, area).

## Dataset

- **Source:** Scikit-learn’s `load_breast_cancer()` dataset
- **Samples:** 569
- **Features:** 30 numeric features
- **Target:** 0 = Malignant, 1 = Benign

##  Project Workflow

1. **Data Exploration & Visualization**
   - Summary statistics
   - Pairplots, boxplots, heatmap
   - Feature correlation with target

2. **Preprocessing**
   - StandardScaler for normalization
   - Train/test split

3. **Model Architecture**
   - 3-layer feedforward neural network:
     - Input: 30 features
     - Hidden layers: 16 → 8 neurons (ReLU)
     - Output: 1 neuron (Sigmoid)

4. **Training**
   - Loss: Binary Cross Entropy (BCELoss)
   - Optimizer: Adam
   - 120 epochs

5. **Evaluation**
   - Accuracy score
   - Optionally confusion matrix

6. **Prediction**
   - Predict outcome on new or existing samples

##  Model Performance

- Training Accuracy: **~98%**
- Model shows strong ability to distinguish benign vs malignant tumors

## 🚀 Tools & Libraries

- Python, NumPy, Pandas, Seaborn, Matplotlib
- PyTorch
- scikit-learn


