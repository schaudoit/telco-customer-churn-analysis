# 📊 Telco Customer Churn Analysis

This project aims to analyze customer churn behavior using open data from a telecommunications company.  
It demonstrates my ability to explore business-oriented datasets, build data pipelines, and extract actionable insights.

---

## 🔍 Objectives

- Identify key factors that influence customer churn.
- Visualize customer behavior and segmentations.
- Build predictive models to anticipate churn.
- Provide business recommendations to reduce churn rate.

---

## 📁 Data Source

- [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🧠 Skills & Tools

- Data cleaning and preprocessing (`pandas`, `numpy`)
- Data visualization (`matplotlib`, `seaborn`, `plotly`)
- Machine learning models (`scikit-learn`)
- Model evaluation and business interpretation
- Python scripting and notebook-based analysis

---

## 📈 Results and Model Performance

The final predictive models achieved the following performance:

- **Logistic Regression**:  
  - Accuracy: ~78%  
  - F1-score (Churners): ~0.55  
- **Random Forest**:  
  - Accuracy: ~78%  
  - F1-score (Churners): ~0.54  
- **Decision Tree** (for interpretation):  
  - Depth-limited tree with clear rules to guide business insights

Although model performance was modest in identifying churners, the models helped isolate key factors that influence churn.

---

## 💡 Business Recommendations

Based on the exploratory and predictive analyses, we suggest:

- Focus retention efforts on customers with **month-to-month contracts** and **high monthly charges**.
- Provide proactive support to users without **OnlineSecurity** or **TechSupport**, as these features are linked to lower churn.
- Target marketing actions for new customers with **low tenure** and **Fiber optic internet**, who are at higher risk of churning.
- Consider integrating model predictions into **CRM dashboards** for near real-time churn flagging, while being cautious of predictive limitations.

---

## 📦 Project Structure
# Telco Customer Churn Analysis

This project analyzes customer churn behavior using open-source data from a telecommunications company. It showcases a complete data science pipeline—from data cleaning to machine learning modeling and business recommendations.

---

## Objectives

- Identify the main factors contributing to customer churn.
- Visualize customer behavior and segment patterns.
- Build predictive models to anticipate churn.
- Generate actionable business insights to mitigate churn.

---

## Data Source

- [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Skills & Tools

- **Languages & Libraries**: Python, pandas, numpy, matplotlib, seaborn, plotly, scikit-learn
- **Tasks Covered**:
  - Data cleaning and preprocessing
  - Exploratory Data Analysis (EDA)
  - Feature engineering and correlation analysis
  - Machine learning modeling and evaluation
  - Dashboard and business interpretation

---

## Project Structure

```
project2_churn/
│
├── data/                   # Original and enriched datasets
│   └── telco_churn.csv
│
├── notebooks/              # Jupyter Notebook with final analysis
│   └── churn_analysis.ipynb
│
├── scripts/                # Python scripts for modular pipeline
│   └── churn_pipeline.py
│
├── requirements.txt        # Environment dependencies
├── .gitignore              # Git tracking exclusions
└── README.md               # Project overview
```