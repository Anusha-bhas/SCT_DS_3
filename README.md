# Bank Marketing Data: Preprocessing for Classification

This repository contains a Jupyter Notebook (`skillcraft-3.ipynb`) that outlines the initial steps for **preprocessing and preparing a bank marketing dataset** for a machine learning classification task. The primary objective is to predict whether a client will subscribe to a term deposit (`y`).

## 📊 Initial Data State

The dataset contains **4,521 samples** and **17 features**. The target variable (`y`) exhibits **class imbalance**:

* **No** (Did not subscribe): ~88.5%
* **Yes** (Subscribed): ~11.5%

## ⚙️ Data Transformation

To prepare the data for a machine learning model, the following transformations were performed:

* **Target Variable Encoding:** The original categorical target variable (`y`) was converted to a **binary numeric format** where **'yes' is 1** and **'no' is 0**.
* **One-Hot Encoding:** All **nominal categorical features** (columns with `object` dtype) were converted into numerical features using **One-Hot Encoding** (`pd.get_dummies()`). The shape of the dataset increased from `(4521, 17)` to `(4521, 43)` after encoding.

## 🧪 Model Split

The final feature set (`X`) and the encoded target (`y`) were split into training and testing subsets:

* **Training Set (80%):** 3,616 samples
* **Testing Set (20%):** 905 samples

The split used **stratified sampling** (`stratify=y`) to ensure both the training and testing sets maintain the original **imbalanced distribution** of the target variable.