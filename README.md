Project Name : Customer Churn Prediction & Risk Segmentation Dashboard
Project Overview :
 Subscription-based businesses lose significant revenue every year due to customer churn — customers who cancel or stop using a service.
 This project builds a complete end-to-end Machine Learning pipeline that:

 Explores and analyzes customer data (EDA)
 Cleans and engineers meaningful features
 Trains and compares 3 classification models
 Segments customers into High / Medium / Low churn risk tiers
 Visualizes results in clear, business-friendly charts
 Translates model findings into actionable business decisions


Problem Statement :
 Which customers are most likely to cancel their subscription — and why?
 Telecom companies face intense competition. Switching services is easy, and price-sensitive customers leave quickly. Predicting churn before it happens allows the business to:

 > Offer targeted discounts to at-risk customers
 > Design better onboarding for new customers
 > Prioritize customer success resources efficiently

DATASET :
PropertyDetails :
Source : IBM Telco Customer Churn — KaggleFileWA_Fn-UseC_-Telco-Customer-Churn.csv
Rows : 7,043 customers
Columns : 21 features
TargetChurn : (Yes / No)
Churn Rate :~26.5%

Key Features Include:

Demographics: gender, SeniorCitizen, Partner, Dependents
Account Info: tenure, Contract, MonthlyCharges, TotalCharges
Services: InternetService, TechSupport, StreamingTV, OnlineSecurity


How to Run
Step 1 — Clone the Repository
git clone https://github.com/your-username/ChurnAnalysis.git
cd ChurnAnalysis

Step 2 — Install Dependencies
pip install -r requirements.txt

Step 3 — Launch Jupyter Notebook
jupyter notebook

Step 4 — Open and Run
Open analysis.ipynb and run all cells from top to bottom.
Make sure WA_Fn-UseC_-Telco-Customer-Churn.csv is in the same folder as the notebook before running.


Project Workflow :
Raw CSV Data
     │
     ▼
Task 1 — EDA (shape, nulls, class imbalance, heatmap)
     │
     ▼
Task 2 — Preprocessing (fix TotalCharges, encode, scale, split)
     │
     ▼
Task 3 — Train 3 Models → Compare → Tune Best Model
     │
     ▼
Task 4 — Risk Segmentation (High / Medium / Low tiers)
     │
     ▼
Task 5 — 5 Visualizations (feature importance, contracts, tenure, donut, Plotly)
     │
     ▼
Task 6 — Business Insights & Recommendations


> XGBoost (Tuned) was selected as the final model.
Hyperparameter tuning via RandomizedSearchCV improved ROC-AUC from 0.8249 → 0.8475

Key Insights

Tenure is the strongest predictor — Customers in their first 0–10 months churn at dramatically higher rates. Once a customer passes 12 months, loyalty increases significantly.
Month-to-Month contracts are the #1 risk factor — These customers churn at 3–4× the rate of annual or two-year contract holders. Zero lock-in = zero friction to leave.
Higher monthly charges → higher churn — Premium-paying customers have higher expectations. When service quality doesn't match the price, they leave first.


Business Recommendations
1. Loyalty Conversion Campaign
Target all High Risk customers on Month-to-Month contracts with a discounted offer to upgrade to a 1-Year plan.

Priority: tenure < 12 months AND monthly charges > $65
Even a 10–15% discount is cheaper than losing the customer entirely

2. New Customer Onboarding Program
Since churn peaks in the first 0–10 months, build a structured onboarding journey:

Proactive check-in at Day 30 and Day 60
Personalized tips based on the customer's plan
Loyalty reward (e.g., free month or upgrade) at the 6-month mark
Feedback survey at Day 45 to catch dissatisfaction early

Visualizations :
ChartDescription :
> feature_importance.png :Top 10 features driving churn (XGBoost)
> churn_by_contract.png :Churn rate by contract type
> tenure_distribution.png : churned vs not churned (KDE + histogram)
> risk_tier_donut.png : Customer count per risk tier
> model_comparison.png : ROC curve — all 3 models on one chart
> confusion_matrices.png : Confusion matrix — all 3 models side by side
> risk_tier_analysis.png : Avg charges, tenure & churn rate by tier


 Limitations & Future Work
Current Limitations:

Recall of ~51% means the model misses about half of actual churners
Dataset is a static historical snapshot — needs periodic retraining
No behavioral signals (usage frequency, support tickets, payment delays)

Future Improvements:

 Apply SMOTE to fix class imbalance and improve recall
 Add real-time behavioral features (app usage, complaint history)
 Lower classification threshold from 0.5 → 0.4 to catch more churners
 Deploy as a live weekly scoring pipeline
 Explore deep learning models (TabNet) for higher accuracy

 👤 Author
[keerthana shetty]
Data Science & Analytics
May 2026

License
This project is for educational and internship purposes.
Dataset credit: IBM / BlastChar on Kaggle

