# 🛡️ Fraud Detection Model

A robust, end-to-end machine learning pipeline to detect fraudulent financial transactions using historical data from cardholders, merchants, and transaction activity.

---

## 📊 Dataset Sources

- **Transactions**: Historical transaction logs with timestamps, merchant info, and amounts.
- **Cards & Users**: Metadata including card brand, type, and whether the card has a chip.
- **Labels**: Fraud indicators (`Yes` / `No`) for a subset of transactions.
- **MCC Codes**: Merchant Category Code descriptions.

---

## 🔧 Key Processing Steps

### 🧹 Data Cleaning & Transformation
- Removed currency symbols and formatted numeric fields.
- Converted date columns to proper `datetime` format.
- Handled missing values and standardized inconsistent entries.

### 🏗️ Feature Engineering
- Created `updated_amt` to convert all amounts to positive values.
- Aggregated transaction statistics at the **client-card** level:
  - Mean, max, min, total amount
  - Count of transactions
- Calculated **fraud ratios**:
  - Percentage of fraud by count and amount
- Computed a **risk score (0–3)** based on:
  - High fraud ratio (above 75th percentile thresholds)
  - High absolute fraud amount

### 🔤 Label Encoding
Used `LabelEncoder` to transform categorical variables:
- `card_brand`, `card_type`, `has_chip`
- `merchant_state`, `merchant_city`, `errors`, `zip`

---

## 🤖 Model Training

### 🎯 Preparation
- Stratified sampling on the fraud label to ensure class balance.
- Feature scaling using `StandardScaler`.

### 🧪 Models Used
1. **Random Forest**
2. **XGBoost**
3. **Decision Tree**
4. **K-Nearest Neighbors (KNN)**
5. **Gradient Boosting**
6. **Voting Classifiers**
   - Hard Voting
   - Soft Voting

Each model was tuned with class balancing (`class_weight='balanced'` or `scale_pos_weight`) to handle imbalance in fraud vs. non-fraud cases.

---

## 🧠 Model Evaluation

Models were evaluated on:
- ✅ Accuracy
- 🎯 Precision
- 🔁 Recall
- 📊 F1-Score
- 📈 ROC-AUC Score
- 📉 Matthews Correlation Coefficient (MCC)

### 📌 Results Visualization
- **ROC Curves** for model comparison.
- **Confusion Matrices** for error analysis.
- Top-performing model: **Voting Classifier (Soft Voting)** with high precision and recall.

---

## 📦 Installation & Setup

To run this project locally or on Google Colab:

```bash
# Clone this repository
git clone https://github.com/Raghavcpp/Fraud-Detection-Model.git

# Open the notebook in Google Colab or Jupyter
