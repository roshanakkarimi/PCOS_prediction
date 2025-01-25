# PCOS Prediction using Neural Networks

This project aims to predict Polycystic Ovary Syndrome (PCOS) using a Sequential Neural Network (NN) model. The dataset includes various clinical, hormonal, and metabolic features, and the model is trained to classify individuals as having PCOS or not.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview
Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder affecting individuals of reproductive age. Early detection and diagnosis are crucial for effective management. This project uses a **Sequential Neural Network** to predict PCOS based on clinical and biochemical features.

### Key Features:
- Preprocessing of clinical and biochemical data.
- Training a Sequential Neural Network for binary classification.
- Evaluation of model performance using accuracy, loss, and validation metrics.

---

## Dataset
The dataset used in this project contains the following features:
- **Clinical Features**: Age, Weight, Height, BMI, Blood Group, etc.
- **Hormonal Levels**: FSH, LH, TSH, AMH, Prolactin, etc.
- **Metabolic Markers**: Cholesterol, Triglycerides, HDL, LDL, etc.
- **Symptoms**: Weight gain, hair growth, skin darkening, etc.

The dataset is split into:
- **Training Set**: 70%
- **Validation Set**: 15%
- **Test Set**: 15%

---

## Installation
To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/pcos-prediction-nn.git
   cd pcos-prediction-nn

2. **Set Up a Virtual Environment (Optional but Recommended)**:
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies**:
   ```bash
    pip install -r requirements.txt

4. **Usage**

Preprocess the Data:
Run the preprocessing script to clean and normalize the data:
    ```bash 
    
    python main.py

Train the Model:
Train the Sequential Neural Network:

  ```bash
  python model.py
