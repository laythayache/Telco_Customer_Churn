# 0. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Prepare output file (now in UTF-8)
log = open('analysis.txt', 'w', encoding='utf-8')

# 2. Load & initial inspect
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
log.write("1. RAW DATA INSPECTION\n")
log.write(f"Shape: {df.shape}\n\n")
log.write("Info:\n")
df.info(buf=log)
log.write("\nHead:\n")
log.write(df.head().to_string())
log.write("\n\n")

# 3. Missing‐value summary (before cleaning)
log.write("2. MISSING VALUES (per column)\n")
log.write(df.isnull().sum().to_string())
log.write("\n\n")

# 4. Clean & drop ID
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
missing_tc = df['TotalCharges'].isnull().sum()
log.write(f"3. TotalCharges missing before impute: {missing_tc}\n")
median_tc = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_tc)
df.drop('customerID', axis=1, inplace=True)
log.write("Dropped customerID, imputed TotalCharges, no more missing.\n\n")

# 5. Encode target & categoricals
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
cat_cols = df.select_dtypes(include='object').columns.tolist()
log.write(f"4. CATEGORICAL FEATURES ONE-HOT ENCODED: {cat_cols}\n\n")
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 6. Scale numeric features
num_cols = df.select_dtypes(include=[np.number]).columns.drop('Churn')
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
log.write(f"5. NUMERIC FEATURES SCALED: {list(num_cols)}\n\n")

# 7. Check class imbalance
churn_counts = df['Churn'].value_counts()
log.write("6. CHURN DISTRIBUTION\n")
log.write(churn_counts.to_string())
log.write("\n\n")
plt.figure()
churn_counts.plot(kind='bar')
plt.xticks([0,1], ['Stayed','Churned'], rotation=0)
plt.title("Churn imbalance")
plt.show()

# 8. Feature distributions
log.write("7. FEATURE DISTRIBUTIONS\n")
for col in num_cols:
    desc = df[col].describe()
    log.write(f"{col} (numeric):\n{desc.to_string()}\n\n")
    plt.figure()
    df[col].hist(bins=30)
    plt.title(f'{col} Distribution')
    plt.show()

cat_dummies = [c for c in df.columns if c not in num_cols and c!='Churn']
for col in cat_dummies:
    counts = df[col].value_counts()
    log.write(f"{col} (binary cat):\n{counts.to_string()}\n\n")
    plt.figure()
    counts.plot(kind='bar')
    plt.xticks(rotation=45)
    plt.title(f'Counts of {col}')
    plt.show()

# 9. Churn vs feature comparison
log.write("8. CHURN VS FEATURE\n")
for col in num_cols:
    grp = df.groupby('Churn')[col].describe()
    log.write(f"{col} by Churn:\n{grp.to_string()}\n\n")
    plt.figure()
    sns.boxplot(x='Churn', y=col, data=df)
    plt.title(f'{col} by Churn')
    plt.show()

for col in cat_dummies:
    rates = df.groupby(col)['Churn'].mean()
    log.write(f"Churn rate by {col}:\n{rates.to_string()}\n\n")
    plt.figure()
    rates.plot(kind='bar')
    plt.title(f'Churn rate by {col}')
    plt.ylabel('Churn rate')
    plt.show()
'''
# 10. Correlation heatmap and values
corr = df.corr()
log.write("9. CORRELATION MATRIX (top correlations with Churn)\n")
churn_corr = corr['Churn'].sort_values(ascending=False)
log.write(churn_corr.to_string())
log.write("\n\nFull correlation matrix:\n")
log.write(corr.to_string())
log.write("\n\n")
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label':'corr'})
plt.title("Feature Correlation Matrix")
plt.show()
'''
# 11. Baseline confusion matrix
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0,1])

log.write("10. BASELINE CONFUSION MATRIX (DummyClassifier)\n")
log.write("Labels [Stayed, Churned]\n")
log.write(np.array2string(cm))
log.write("\n\n")
disp = ConfusionMatrixDisplay(cm, display_labels=['Stayed','Churned'])
disp.plot()
plt.title("Baseline DummyClassifier Confusion Matrix")
plt.show()

# 12. Finish up
log.close()
print("Analysis complete—see analysis.txt for detailed summaries.")
# Note: The code above is a complete script for data analysis and does not require any additional imports or modifications.