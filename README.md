# 📧 Email Marketing Campaign Success Prediction (Python)

[![Python](https://img.shields.io/badge/Built%20With-Python-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

---

## 📘 Project Overview

This project implements a **comprehensive machine learning pipeline** to predict email campaign success for a **skin care clinic** marketing campaign. The analysis utilizes **proper train/test splitting**, **extensive visualizations**, and compares **four different classification algorithms** to identify which customers are most likely to open marketing emails.

**Key Features:**
- ✅ **Train/Test Split (80/20)** for robust model evaluation
- ✅ **Statistical Significance Testing** using statsmodels
- ✅ **Multiple Classification Algorithms** (Logistic Regression, Naive Bayes, SVM)
- ✅ **Comprehensive Visualizations** (EDA, ROC Curves, Confusion Matrices)
- ✅ **ROC-AUC Analysis** for model comparison
- ✅ **Production-Ready Models** with persistence and metadata tracking

**Dataset:** 683 customer records with demographics, purchase recency, billing history, and email response data.

---

## 🎯 Business Problem

The skin care clinic needs to **optimize email marketing campaigns** by:
- Identifying customers most likely to engage with email communications
- Understanding which customer characteristics drive email engagement
- Reducing marketing waste by targeting high-probability customers
- Improving ROI on marketing spend through data-driven targeting

---

## 📊 Dataset Description

| Variable           | Type    | Description                                              |
|--------------------|---------|----------------------------------------------------------|
| `Success`          | Binary  | Email opened (1) or not opened (0) - **TARGET**          |
| `Gender`           | Integer | 1 = Male, 2 = Female                                     |
| `AGE`              | Category| Age group: <=30, <=45, <=55, >55                         |
| `Recency_Service`  | Integer | Days since last service purchase                         |
| `Recency_Product`  | Integer | Days since last product purchase                         |
| `Bill_Service`     | Float   | Total service billing (last 3 months)                    |
| `Bill_Product`     | Float   | Total product billing (last 3 months)                    |

**Response Rate:** ~28% email open rate (baseline)
**Train/Test Split:** 80/20 stratified split maintaining class distribution

---

## 🔬 Comprehensive Methodology

### 1️⃣ **Exploratory Data Analysis (EDA)**
- Distribution analysis of all variables
- Success rate analysis by demographics (Gender, Age)
- Billing and recency pattern exploration
- Correlation matrix with heatmap visualization
- **9 Visualizations** covering all aspects of the data

### 2️⃣ **Data Preparation**
- Label encoding for categorical AGE variable
- Train/Test split (80/20) with stratification
- Feature engineering for combined variables
- Data validation and quality checks

### 3️⃣ **Question 1: Binary Logistic Regression**
- Statistical analysis using **statsmodels** for p-values
- Identification of significant predictors (p < 0.05)
- Coefficient interpretation (positive/negative effects)
- Model training on significant variables only
- **Visualization:** Coefficient plot showing significance

### 4️⃣ **Question 2: Naive Bayes Classification**
- Gaussian Naive Bayes with all features
- Probability estimation for email opening
- Performance comparison with Logistic Regression
- Classification report with precision, recall, F1-score

### 5️⃣ **Question 3: Feature Engineering & SVM**
- Created combined features:
  - `Total_Recency` = Service + Product recency
  - `Total_Bill` = Service + Product billing
  - `Recency_Ratio` = Service/Product recency ratio
  - `Bill_Ratio` = Service/Product billing ratio
- Logistic Regression with engineered features
- Support Vector Machine (RBF kernel) implementation
- Advanced model comparison

### 6️⃣ **Model Evaluation & Comparison**
- **ROC Curves** for all four models
- **AUC Score** comparison
- **Confusion Matrices** visualization
- **Classification Reports** with detailed metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Comprehensive performance summary table

---

## 📈 Model Performance Results

### Performance Summary Table

| Model                              | Features | AUC    | Accuracy | Precision | Recall | F1-Score |
|------------------------------------|----------|--------|----------|-----------|--------|----------|
| Logistic Regression (Significant)  | 4-5      | ~0.75  | ~0.72    | ~0.55     | ~0.65  | ~0.60    |
| Logistic Regression (Combined)     | 6        | ~0.74  | ~0.71    | ~0.54     | ~0.63  | ~0.58    |
| Naive Bayes (All Variables)        | 6        | ~0.73  | ~0.70    | ~0.52     | ~0.68  | ~0.59    |
| SVM (RBF Kernel)                   | 6        | ~0.72  | ~0.69    | ~0.50     | ~0.60  | ~0.55    |

*Note: Exact values depend on random state and specific dataset split*

### 🏆 Best Model: **Logistic Regression (Significant Variables)**

**Why it's the best:**
- ✅ Highest AUC score (~0.75)
- ✅ Interpretable coefficients
- ✅ Fast training and prediction
- ✅ Uses only statistically significant predictors
- ✅ Production-ready and explainable

### Key Findings

**Significant Predictors (p < 0.05):**
1. **Bill_Service** (Positive) - Higher service spending increases email open probability
2. **Bill_Product** (Positive) - Product purchases correlate with engagement
3. **Recency_Service** (Negative) - Recent service visits increase open rates
4. **Recency_Product** (Negative) - Recent product purchases increase engagement

---

## 📂 Project Structure

```
email-campaign-prediction/
├── data/
│   ├── raw/
│   │   └── Email Campaign.csv
│   ├── processed/
│   │   └── email_campaign_processed.csv
│   └── predictions/
├── models/
│   ├── logistic_regression_significant.pkl
│   ├── naive_bayes_model.pkl
│   ├── logistic_regression_combined.pkl
│   ├── svm_model.pkl
│   ├── age_label_encoder.pkl
│   ├── significant_variables.pkl
│   └── model_metadata.json
├── reports/
│   ├── figures/
│   │   ├── 01_exploratory_data_analysis.png
│   │   ├── 02_correlation_matrix.png
│   │   ├── 03_train_test_split.png
│   │   ├── 04_logistic_regression_coefficients.png
│   │   ├── 05_comprehensive_model_comparison.png
│   │   └── 06_confusion_matrices.png
│   └── model_performance_summary.csv
├── notebooks/
│   └── 01_complete_analysis.ipynb
├── environment/
│   ├── requirements.txt
│   └── environment.yml
├── main.py
├── load_and_predict.py
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip or conda package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/kochezz/ML1_Marketing_Campaign_PY.git
cd email-campaign-prediction
```

2. **Create virtual environment**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda env create -f environment/environment.yml
conda activate email-campaign
```

3. **Install dependencies**
```bash
pip install -r environment/requirements.txt
```

### Required Libraries

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
```

---

## 💻 Usage

### Run Complete Analysis

```python
# Execute the full pipeline
python main.py
```

This will:
- ✅ Load and explore data with visualizations
- ✅ Create train/test splits (80/20)
- ✅ Train all 4 models
- ✅ Generate comprehensive visualizations:
  - Exploratory data analysis (9 plots)
  - Correlation matrix
  - Train/test distribution
  - Coefficient significance plot
  - ROC curves (3 comparison plots)
  - Confusion matrices (4 models)
- ✅ Save all models to `/models` directory
- ✅ Save performance metrics and metadata
- ✅ Generate performance summary CSV

### Load Models for Predictions

```python
import joblib
import pandas as pd

# Load best model
model = joblib.load('models/logistic_regression_significant.pkl')
age_encoder = joblib.load('models/age_label_encoder.pkl')
significant_vars = joblib.load('models/significant_variables.pkl')

# Prepare new data
new_data = pd.read_csv('new_customers.csv')
new_data['AGE_Encoded'] = age_encoder.transform(new_data['AGE'])
X_new = new_data[significant_vars]

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

# Add to dataframe
new_data['Prediction'] = predictions
new_data['Open_Probability'] = probabilities
```

### Quick Prediction Example

```python
from load_and_predict import predict_with_best_model

# Predict on new customer data
results = predict_with_best_model('data/new_customers.csv')

# View results
print(results[['Gender', 'AGE', 'Will_Open_Email', 'Probability']].head())

# Save predictions
results.to_csv('predictions_output.csv', index=False)
```

---

## 📊 Visualizations Generated

### 1. Exploratory Data Analysis (EDA)
**File:** `01_exploratory_data_analysis.png`
- 9 comprehensive plots showing:
  - Target variable distribution
  - Gender and age distributions
  - Success rates by demographics
  - Recency and billing distributions

### 2. Correlation Matrix
**File:** `02_correlation_matrix.png`
- Heatmap showing relationships between all variables
- Identifies which features correlate with email success

### 3. Train/Test Split Validation
**File:** `03_train_test_split.png`
- Confirms proper stratification
- Shows class balance in both sets

### 4. Logistic Regression Coefficients
**File:** `04_logistic_regression_coefficients.png`
- Visual representation of coefficient magnitudes
- Color-coded by statistical significance

### 5. Comprehensive Model Comparison
**File:** `05_comprehensive_model_comparison.png`
- 4 subplots showing:
  - LR vs NB ROC curves
  - LR-Combined vs SVM ROC curves
  - All models together
  - AUC score bar chart

### 6. Confusion Matrices
**File:** `06_confusion_matrices.png`
- 4 confusion matrices for all models
- Shows true positives, false positives, etc.

---

## 💾 Model Persistence

All trained models are saved using `joblib` for production deployment:

```
models/
├── logistic_regression_significant.pkl  # Best model ⭐
├── naive_bayes_model.pkl
├── logistic_regression_combined.pkl
├── svm_model.pkl
├── age_label_encoder.pkl               # For preprocessing
├── significant_variables.pkl            # Feature names
└── model_metadata.json                  # Complete metadata
```

### Model Metadata (`model_metadata.json`)

Contains comprehensive information:
- Training date and dataset statistics
- Train/test split details and response rates
- Feature lists for each model
- Complete performance metrics (AUC, Accuracy, Precision, Recall, F1)
- Best model identifier

---

## 📌 Key Insights & Recommendations

### Business Insights

1. **Customer Engagement Drivers:**
   - Higher spending customers (both service and product) are more likely to open emails
   - Recent purchasers show higher engagement rates
   - Recency is more important than purchase frequency

2. **Targeting Strategy:**
   - Focus email campaigns on customers with recent purchases
   - Prioritize high-value customers (high billing amounts)
   - Consider re-engagement campaigns for customers with long recency

3. **Expected Performance:**
   - Model can identify ~75% of potential email openers (AUC = 0.75)
   - Precision of ~55% means about half of predicted openers will actually open
   - Recall of ~65% means we'll capture 65% of actual openers

### Technical Recommendations

1. **Model Deployment:**
   - Use Logistic Regression (Significant) for production
   - Implement probability thresholds based on campaign goals
   - Set up A/B testing to validate model performance

2. **Monitoring:**
   - Track model performance over time
   - Retrain quarterly with new data
   - Monitor for concept drift

3. **Future Improvements:**
   - Collect more features (email open history, device type, time preferences)
   - Experiment with ensemble methods
   - Implement customer segmentation for targeted strategies

---

## 📌 Next Steps

- **🌐 API Development:** Create REST API using Flask/FastAPI for real-time predictions
- **📱 Dashboard:** Build interactive Streamlit dashboard for marketing team
- **🔄 A/B Testing:** Validate model predictions with real campaign tests
- **📈 Model Monitoring:** Implement performance tracking over time
- **🎯 Segmentation:** Apply clustering for more granular customer segments
- **🔁 AutoML Pipeline:** Automated retraining with new campaign data

---

## 🎓 Key Learning Outcomes

✅ End-to-end machine learning project with proper train/test methodology
✅ Statistical significance testing and interpretation
✅ Multiple model comparison using ROC-AUC
✅ Feature engineering and domain knowledge application
✅ Production-ready code with comprehensive documentation
✅ Professional visualization and reporting
✅ Model persistence and deployment readiness

---

## 📖 References

- Hosmer, D.W., Lemeshow, S. (2000). *Applied Logistic Regression*
- James, G., et al. (2013). *An Introduction to Statistical Learning*
- scikit-learn documentation: https://scikit-learn.org/
- statsmodels documentation: https://www.statsmodels.org/

---

## 👨‍💼 Author

**[Your Name]**  
📧 Email: [wphiri@beda.ie]  
🔗 LinkedIn: [[LinkedIn Profile](https://www.linkedin.com/in/william-phiri-866b8443/)]  
🐙 GitHub: [Kochezz](https://github.com/kochezz)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset provided for academic assignment
- Built as part of Level 9 NFQ Data Science coursework
- Thanks to instructors and peers for feedback

---

**⭐ If you found this project helpful, please consider giving it a star!**
