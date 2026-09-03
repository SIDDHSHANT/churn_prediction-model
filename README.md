# 📉 Customer Churn Prediction using PyTorch ANN

This project uses **Artificial Neural Networks (ANN)** with **PyTorch** to predict whether a customer is likely to leave a company.

The model is trained on customer information such as credit score, geography, gender, age, balance, number of products, estimated salary, and other customer-related features.

The target variable is **`Exited`**, where:

* `0` → Customer stays
* `1` → Customer leaves

---

## 📌 Project Overview

Customer churn prediction is a classification problem where the goal is to identify customers who are likely to leave a service.

This project demonstrates a complete Deep Learning workflow:

```text
Customer Dataset
       ↓
Data Cleaning
       ↓
Feature Selection
       ↓
Categorical Encoding
       ↓
Train/Test Split
       ↓
Feature Scaling
       ↓
Convert Data to PyTorch Tensors
       ↓
DataLoader
       ↓
Build ANN
       ↓
Train Model
       ↓
Evaluate Model
       ↓
Plot Training & Validation Loss
```

---

## 📂 Dataset

The project uses:

```text
Churn_Modelling.csv
```

The dataset contains customer information and an `Exited` column that represents customer churn.

### Important Features

Some of the features used by the model include:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary

The following columns are removed because they are identifiers or unnecessary for prediction:

```text
RowNumber
CustomerId
Surname
```

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

### 1. Remove unnecessary columns

```python
drop_cols = ["RowNumber", "CustomerId", "Surname"]
```

### 2. Encode categorical features

`Geography` is converted using one-hot encoding:

```python
pd.get_dummies()
```

`Gender` is converted using:

```python
LabelEncoder()
```

### 3. Separate features and target

```python
X = data.drop(columns=["Exited"])
y = data["Exited"]
```

### 4. Train/Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

### 5. Feature Scaling

`StandardScaler` is used to standardize the input features.

---

## 🧠 ANN Architecture

The project uses a simple Artificial Neural Network built with PyTorch.

```text
Input Layer
     ↓
Linear Layer
Input → 64 neurons
     ↓
ReLU
     ↓
Linear Layer
64 → 32 neurons
     ↓
ReLU
     ↓
Linear Layer
32 → 1 neuron
     ↓
Sigmoid
     ↓
Churn Prediction
```

### Model Configuration

| Parameter         |      Value |
| ----------------- | ---------: |
| Hidden Layer 1    | 64 neurons |
| Hidden Layer 2    | 32 neurons |
| Activation        |       ReLU |
| Output            |   1 neuron |
| Output Activation |    Sigmoid |
| Optimizer         |       Adam |
| Epochs            |        200 |
| Batch Size        |         32 |

---

## 🔥 Training

The model is trained using PyTorch.

The training process:

1. Loads batches using `DataLoader`
2. Performs forward propagation
3. Calculates loss
4. Performs backpropagation
5. Updates model weights using Adam
6. Calculates validation loss
7. Saves the model with the best validation loss

The best model is saved as:

```text
best_model.pt
```

---

## 📊 Evaluation

The trained model is evaluated using classification accuracy.

The prediction probability is converted into a binary prediction using a threshold of `0.5`:

```text
Probability >= 0.5 → Customer Churn
Probability < 0.5  → Customer Stays
```

The project also plots:

* Training Loss
* Validation Loss

This helps visualize how the model learns over the training epochs.

---

## 🛠️ Technologies & Libraries

### Python

Main programming language.

### Pandas

Used for:

* Loading CSV data
* Data manipulation
* Data inspection

### NumPy

Used for numerical operations.

### Scikit-learn

Used for:

* `LabelEncoder`
* `OneHotEncoder`
* `train_test_split`
* `StandardScaler`

### PyTorch

Used for:

* Building the ANN
* Tensor operations
* `TensorDataset`
* `DataLoader`
* Model training
* Optimization

### Matplotlib

Used to visualize training and validation loss.

### Jupyter Notebook

Used for developing and running the project.

---

## 📦 Installation

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn torch jupyter
```

Or create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
scikit-learn
torch
jupyter
```

Then install:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
churn_pred.ipynb
```

Make sure the dataset is available in the project directory:

```text
Churn_Modelling.csv
```

Run the notebook cells from top to bottom.

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── churn_pred.ipynb
├── Churn_Modelling.csv
├── best_model.pt
├── requirements.txt
└── README.md
```

---

## 🎯 Learning Objectives

This project demonstrates:

* Data preprocessing
* Categorical encoding
* Feature scaling
* Train/test splitting
* PyTorch tensors
* DataLoader
* Artificial Neural Networks
* Forward propagation
* Backpropagation
* Adam optimization
* Binary classification
* Model evaluation
* Loss visualization
* Saving PyTorch models

---

## 🚀 Future Improvements

* Add precision, recall, and F1-score
* Generate a confusion matrix
* Add validation data separate from the test set
* Perform hyperparameter tuning
* Experiment with different ANN architectures
* Add dropout for regularization
* Create a Streamlit interface
* Deploy the model as a web application
* Add real-time customer churn prediction

---

## ⚠️ Note

This project is intended for **educational and demonstration purposes**. Predictions from this model should not be treated as definitive business decisions without appropriate validation and domain analysis.

---

## 👨‍💻 Author ->

**Siddhartha Singh**

If you found this project useful, consider giving the repository a ⭐ on GitHub.
