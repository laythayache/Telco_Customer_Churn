# Telco_Customer_Churn

# Project Overview

This project implements a complete pipeline for predicting customer churn using the Telco Customer Churn dataset, encompassing exploratory analysis, data preprocessing, baseline evaluation, and comparative modeling.

---

## Dataset: WA\_Fn-UseC\_-Telco-Customer-Churn.csv

* **Records:** 7,043 customers
* **Features (21 total):**

  * Numeric: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
  * Categorical: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
* **Missing Values:** 11 entries in `TotalCharges` (0.16%), imputed with median
* **Churn Rate:** 26.5% churned, 73.5% retained fileciteturn1file7

---

## data\_interperter.py

Performs data loading, initial exploration, preprocessing, and baseline evaluation:

1. **Data Ingestion & Type Conversion**

   * Reads raw CSV, converts `TotalCharges` to float, drops `customerID`
2. **Missing-Value Imputation**

   * Applies median imputation for `TotalCharges`
3. **Feature Encoding & Scaling**

   * Binarizes target `Churn` (0/1)
   * One-hot encodes categorical features
   * Standardizes numeric features using z-score scaling
4. **Exploratory Data Analysis**

   * Visualizes feature distributions and class imbalance
   * Computes summary statistics stratified by churn label
5. **Baseline Model**

   * Implements `DummyClassifier(strategy="most_frequent")`
   * Records baseline accuracy (\~73.4%), precision, recall, and confusion matrix to `analysis.txt` fileciteturn1file9

---

## data\_train.py

Implements preprocessing pipelines, model training with SMOTE, and evaluation of multiple classifiers:

1. **Preprocessing Pipeline**

   * **Numeric Scaling:** `StandardScaler` on `tenure`, `MonthlyCharges`, and `TotalCharges` to center and scale features.
   * **Dimensionality Reduction:** `PCA(n_components=2)` applied to billing features (`MonthlyCharges`, `TotalCharges`) to capture ≥95% variance and reduce noise.
   * **Categorical Encoding:** One-hot encoding of all categorical variables via `ColumnTransformer`.
   * Pipeline is constructed with scikit-learn’s `Pipeline` and `ColumnTransformer` to ensure transformations occur within cross-validation folds.

2. **Class Imbalance Handling**

   * **SMOTE Oversampling:** Synthetic Minority Oversampling Technique applied to the training fold only, to balance the churned vs. retained classes and prevent data leakage.

3. **Model Definitions, Implementation, and Analysis**

   * **Logistic Regression (L1 & L2)**

     * **What it is:** A linear classifier that models the log-odds of churn as a weighted sum of features.
     * **How it works:** Fits coefficients by maximizing penalized likelihood; L1 (sparse solutions) and L2 (shrinkage).
     * **Usage in code:** `LogisticRegression(penalty='l1', solver='liblinear', C=1.0)` and `penalty='l2'` variants within the pipeline.
     * **Result:** Accuracy \~0.726, ROC AUC \~0.829.
     * **Interpretation:** Strong baseline due to convex optimization; performance limited by linear decision boundary, cannot capture complex interactions.

   * **Random Forest**

     * **What it is:** An ensemble of decision trees trained on bootstrap samples with feature bagging to reduce variance.
     * **How it works:** Aggregates predictions of 100 unpruned trees; each tree votes.
     * **Usage in code:** `RandomForestClassifier(n_estimators=100, bootstrap=True)` after SMOTE in the pipeline.
     * **Result:** Accuracy \~0.737, ROC AUC \~0.789.
     * **Interpretation:** Captures non-linear relationships, but default depth and tree count may underfit subtle patterns; variance-reduction comes at cost of bias.

   * **XGBoost**

     * **What it is:** An optimized implementation of gradient boosting that builds sequential trees to correct predecessor errors.
     * **How it works:** Fits 100 boosting rounds with learning\_rate=0.1 and max\_depth=6 to minimize log loss.
     * **Usage in code:** `XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)` integrated in pipeline.
     * **Result:** Accuracy \~0.752, ROC AUC \~0.819.
     * **Interpretation:** Excellent at capturing complex patterns; performance constrained by default regularization parameters and potential overfitting on noisy features.

   * **Decision Tree**

     * **What it is:** A single-tree classifier that splits on feature thresholds to partition the dataset.
     * **How it works:** Uses Gini impurity to choose splits, limited to max\_depth=5 to control complexity.
     * **Usage in code:** `DecisionTreeClassifier(criterion='gini', max_depth=5)` within the SMOTE pipeline.
     * **Result:** Accuracy \~0.707, ROC AUC \~0.656.
     * **Interpretation:** Underfits due to shallow tree; captures only coarse decision rules, leading to low discrimination.

   * **AdaBoost**

     * **What it is:** A boosting ensemble that sequentially reweights misclassified samples and combines weak learners.
     * **How it works:** 50 iterations of `DecisionTreeClassifier(max_depth=1)` with `learning_rate=1.0`.
     * **Usage in code:** `AdaBoostClassifier(n_estimators=50, learning_rate=1.0, base_estimator=DecisionTreeClassifier(max_depth=1))`.
     * **Result:** Accuracy \~0.725, ROC AUC \~0.833.
     * **Interpretation:** Focus on hard examples improves boundary detection; shallow base learners limit capture of feature interactions.

   * **Gradient Boosting (GBM)**

     * **What it is:** Sequential tree boosting similar to XGBoost but native to scikit-learn.
     * **How it works:** 100 trees, learning\_rate=0.1, max\_depth=3; minimizes deviance loss.
     * **Usage in code:** `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)`.
     * **Result:** Accuracy \~0.747, ROC AUC \~0.839.
     * **Interpretation:** Balances bias-variance effectively; slight underperformance vs. XGBoost due to less optimized implementation.

   * **Gaussian Naive Bayes**

     * **What it is:** A probabilistic classifier assuming feature independence and Gaussian-distributed inputs.
     * **How it works:** Computes posterior probabilities with Bayes’ theorem using mean and variance per class.
     * **Usage in code:** `GaussianNB(var_smoothing=1e-9)` after preprocessing.
     * **Result:** Accuracy \~0.679, ROC AUC \~0.771, recall \~0.869.
     * **Interpretation:** High recall by flagging many churners; independence assumption leads to many false positives and low precision.

   * **Support Vector Machine (SVC)**

     * **What it is:** A max-margin classifier that finds the hyperplane maximizing separation, using an RBF kernel for non-linearity.
     * **How it works:** Solves quadratic optimization; `class_weight='balanced'` adjusts penalties.
     * **Usage in code:** `SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True)`.
     * **Result:** Accuracy \~0.737, ROC AUC \~0.828.
     * **Interpretation:** Captures non-linear boundaries; sensitive to scaling and kernel parameters, which may require tuning for further gains.

4. **Evaluation Metrics**

   * Computes accuracy, precision, recall, F1-score, and ROC AUC for each model.
   * Generates confusion matrices and extracts logistic coefficients for interpretability.

5. **Result Logging**

   * Writes the complete performance table to `model_comparison.txt` for downstream analysis.

---

## Model Performance Summary (from model\_comparison.txt)
**Key Insights:**

* Ensemble methods (Gradient Boosting, XGBoost) achieved the highest ROC AUC, indicating superior discrimination.
* GaussianNB exhibits high recall for churn but low precision, suggesting many false positives.
* Logistic regression provides a good balance of interpretability and performance.

**Conclusion:**
Gradient Boosting offers the best overall performance trade-off (accuracy: 74.7%, F1: 0.618, ROC AUC: 0.839) and is recommended for deployment when optimizing both predictive power and interpretability via SHAP analysis.
