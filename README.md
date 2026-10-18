#  Customer Churn Prediction: A Recall-Optimized Machine Learning Approach

##  Project Overview
Customer churn is a critical challenge for subscription-based businesses. This project builds an end-to-end Machine Learning pipeline to predict which customers are most likely to leave (churn). Because the cost of acquiring a new customer is much higher than retaining an existing one, this model focuses on **maximizing Recall** to ensure the business catches as many at-risk customers as possible before they leave.

##  Business Objective & Success Metrics
The dataset is highly imbalanced (80% stay, 20% churn). Using standard Accuracy is misleading, as a model could predict "No Churn" for everyone and still be 80% accurate. 
*   **Primary Metric (Recall):** Minimizing False Negatives. We want to catch as many actual churners as possible so the retention team can intervene.
*   **Trade-off:** We accept a lower Precision (more False Positives). Offering a small discount to a happy customer is far cheaper than losing a customer entirely.

##  Methodology & Pipeline
The project follows a robust, leak-proof Data Science pipeline:
1.  **Data Quality & Cleaning:** Handling missing values, correcting categorical inconsistencies (e.g., 'M', 'male' → 'Male'), and removing duplicates.
2.  **Feature Engineering:** Creating new predictive features such as `TotalCharges`, `MonthlyChargesPerAge`, and `HighUsage`. Handling outliers using the IQR method.
3.  **Preprocessing Pipeline:** Using `ColumnTransformer` and `Pipeline` to scale numerical data and one-hot encode categorical data.
4.  **Model Building & Overfitting Check:** Comparing Logistic Regression, Decision Tree, and Random Forest. (Detected significant overfitting in the baseline Random Forest).
5.  **Cross-Validation & Tuning:** Using 5-Fold Stratified CV and `GridSearchCV` to find the optimal hyperparameters and prevent overfitting.
6.  **Threshold Optimization:** Lowering the classification threshold from `0.50` to **`0.35`** to drastically boost Recall. 

##  Key Results

| Model | Default Threshold (0.50) Recall | Optimized Threshold (0.35) Recall |
| :--- | :--- | :--- |
| Logistic Regression | 58% | **94.0%** |
| Decision Tree | 51% | **93.7%** |
| **Random Forest (Tuned)** | 52% | **95.2% (best)** |

**Final Selected Model:** The **Tuned Random Forest** at a **0.35 threshold**. 
*   **Recall:** 95.24% (Catches 380 out of 399 churners).
*   **Precision:** 20.8% 

By deploying this model, the retention team can proactively target approximately 95% of at-risk customers, significantly reducing revenue loss.

##  Technologies Used
*   **Language:** Python 3.12
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn (Pipelines, GridSearchCV, StandardScaler, OneHotEncoder, RandomForest, etc.)
*   **Data Visualization:** Matplotlib, Seaborn
