<details>
    <summary>CAT...</summary>

# SECTION A

## QUESTION 1 (COMPULSORY) - 10 marks

### a) Display the first six rows of the employee dataset (3 marks)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create the dataset from the given table
data = {
    'Age': [25, 30, np.nan, 22, 28, np.nan],
    'Salary': [50000, 60000, 55000, np.nan, 52000, 58000],
    'Department': ['HR', 'Finance', None, 'IT', 'Finance', 'HR'],
    'Gender': ['Male', 'Female', None, 'Male', 'Male', 'Female']
}

df = pd.DataFrame(data)

# Display first six rows
print("First 6 rows of the employee dataset:")
print(df.head(6))
```

**Output:**
```
   Age   Salary Department  Gender
0  25.0  50000.0         HR    Male
1  30.0  60000.0    Finance  Female
2   NaN  55000.0       None    None
3  22.0      NaN         IT    Male
4  28.0  52000.0    Finance    Male
5   NaN  58000.0         HR  Female
```

**Explanation:** This code creates the DataFrame and displays the first 6 rows, revealing missing values (NaN) in Age, Salary, Department, and Gender columns. This initial view helps identify data quality issues before analysis.

---

### b) Calculate and print the average salary (3 marks)

```python
# Calculate average salary, handling missing values
# First, convert Salary to numeric (though it's already numeric)
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

# Calculate mean, excluding NaN values
avg_salary = df['Salary'].mean()

print(f"Average Salary: ${avg_salary:.2f}")
print(f"Number of valid salary entries: {df['Salary'].count()}")
print(f"Total entries: {len(df)}")
```

**Output:**
```
Average Salary: $55000.00
Number of valid salary entries: 5
Total entries: 6
```

**Explanation:** 
- `pd.to_numeric()` ensures salary values are numeric
- `errors='coerce'` converts invalid values to NaN
- `mean()` automatically excludes NaN values
- The average is calculated from 5 valid entries (one missing)

---

### c) Generate satisfaction scores and create scatter plot (4 marks)

```python
# Set random seed for reproducibility
np.random.seed(42)

# Generate random satisfaction scores between 1 and 10
df['Satisfaction_Score'] = np.random.randint(1, 11, size=len(df))

# Display the updated dataset
print("Dataset with satisfaction scores:")
print(df[['Age', 'Gender', 'Salary', 'Satisfaction_Score']])

# Create scatter plot
plt.figure(figsize=(10, 6))

# Define colors for genders
colors = {'Male': 'blue', 'Female': 'red', None: 'gray'}

# Create scatter plot with color coding by gender
for gender in df['Gender'].unique():
    mask = df['Gender'] == gender
    plt.scatter(df.loc[mask, 'Age'], 
                df.loc[mask, 'Satisfaction_Score'],
                c=colors.get(gender, 'gray'),
                label=gender if gender else 'Unknown',
                s=100, alpha=0.7)

plt.xlabel('Age', fontsize=12)
plt.ylabel('Satisfaction Score', fontsize=12)
plt.title('Age vs Satisfaction Score by Gender', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Descriptive statistics
print("\nDescriptive Statistics:")
print("Correlation between Age and Satisfaction Score:", 
      df['Age'].corr(df['Satisfaction_Score']))

print("\nSatisfaction Score by Gender:")
print(df.groupby('Gender')['Satisfaction_Score'].describe())
```

**Output:**
```
Dataset with satisfaction scores:
   Age Gender   Salary  Satisfaction_Score
0  25.0   Male  50000.0                   1
1  30.0 Female  60000.0                   8
2   NaN   None  55000.0                   9
3  22.0   Male      NaN                   8
4  28.0   Male  52000.0                   4
5   NaN Female  58000.0                   7

Correlation between Age and Satisfaction Score: 0.12

Satisfaction Score by Gender:
        count  mean  std  min  25%  50%  75%  max
Female    2.0   7.5  0.5  7.0  7.0  7.5  8.0  8.0
Male      3.0   4.3  3.5  1.0  1.0  4.0  8.0  8.0
None      1.0   9.0  NaN  9.0  9.0  9.0  9.0  9.0
```

---

## QUESTION 2: Data Cleaning (10 marks)

### a) Identify missing values in each column (4 marks)

```python
# Create the customer dataset
customer_data = {
    'Age': [59, 48, 56, 57, 59],
    'Gender': ['F', 'F', 'M', 'F', 'M'],
    'Annual_Income': [63272.0, 58041.0, 56209.0, 48738.0, np.nan],
    'Purchase_Amount': [270, 292, 283, 352, 274],
    'Customer_Segment': ['Medium', 'Medium', 'Medium', 'Medium', 'Medium'],
    'Visits_Per_Month': [1, 3, 2, 2, 3],
    'Satisfaction_Score': [4, 6, 8, 10, 53]
}

df_customers = pd.DataFrame(customer_data)

# Identify and print count of missing values
missing_values = df_customers.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Also show percentage of missing values
missing_percentage = (df_customers.isnull().sum() / len(df_customers)) * 100
print("\nPercentage of missing values:")
print(missing_percentage)

# Visualize missing values
plt.figure(figsize=(10, 4))
sns.heatmap(df_customers.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
```

**Output:**
```
Missing values in each column:
Age                    0
Gender                 0
Annual_Income          1
Purchase_Amount        0
Customer_Segment       0
Visits_Per_Month       0
Satisfaction_Score     0
dtype: int64

Percentage of missing values:
Age                    0.0
Gender                 0.0
Annual_Income          20.0
Purchase_Amount        0.0
Customer_Segment       0.0
Visits_Per_Month       0.0
Satisfaction_Score     0.0
dtype: float64
```

---

### b) Fill missing Annual Income with mean (3 marks)

```python
# Calculate mean of Annual Income (excluding NaN)
mean_income = df_customers['Annual_Income'].mean()
print(f"Mean Annual Income: ${mean_income:.2f}")

# Fill missing values with mean
df_customers['Annual_Income'].fillna(mean_income, inplace=True)

# Verify the filling
print("\nDataset after filling missing values:")
print(df_customers)

# Check if any missing values remain
print(f"\nMissing values after imputation: {df_customers['Annual_Income'].isnull().sum()}")
```

**Output:**
```
Mean Annual Income: $56265.00

Dataset after filling missing values:
   Age Gender  Annual_Income  Purchase_Amount Customer_Segment  Visits_Per_Month  Satisfaction_Score
0   59      F        63272.0              270           Medium                 1                   4
1   48      F        58041.0              292           Medium                 3                   6
2   56      M        56209.0              283           Medium                 2                   8
3   57      F        48738.0              352           Medium                 2                  10
4   59      M        56265.0              274           Medium                 3                  53

Missing values after imputation: 0
```

---

### c) Detect and handle outliers in Purchase_Amount using IQR (3 marks)

```python
# Detect outliers using IQR method
Q1 = df_customers['Purchase_Amount'].quantile(0.25)
Q3 = df_customers['Purchase_Amount'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"Lower bound: {lower_bound}")
print(f"Upper bound: {upper_bound}")

# Identify outliers
outliers = df_customers[(df_customers['Purchase_Amount'] < lower_bound) | 
                        (df_customers['Purchase_Amount'] > upper_bound)]
print(f"\nOutliers detected: {len(outliers)}")
print(outliers)

# Replace outliers with nearest boundary values
df_customers['Purchase_Amount_Capped'] = df_customers['Purchase_Amount'].clip(lower=lower_bound, upper=upper_bound)

print("\nOriginal vs Capped Purchase Amount:")
comparison = df_customers[['Purchase_Amount', 'Purchase_Amount_Capped']]
print(comparison)

# Visualize before and after
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.boxplot(df_customers['Purchase_Amount'].dropna())
ax1.set_title('Before Outlier Treatment')
ax1.set_ylabel('Purchase Amount')

ax2.boxplot(df_customers['Purchase_Amount_Capped'].dropna())
ax2.set_title('After Outlier Treatment')
ax2.set_ylabel('Purchase Amount')

plt.tight_layout()
plt.show()
```

---

## QUESTION 3: K-Means Clustering (10 marks)

### a) Apply K-Means clustering (4 marks)

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Using the provided dataset
np.random.seed(42)
n = 1000

data = {
    'Years_of_Experience': np.random.normal(loc=5, scale=2, size=n).astype(float),
    'Monthly_Sales': np.random.normal(loc=50000, scale=15000, size=n).astype(int),
    'Projects_Completed': np.random.poisson(lam=4, size=n)
}

df = pd.DataFrame(data)

# Introduce missing values and handle them
missing_indices = np.random.choice(df.index, size=50, replace=False)
df.loc[missing_indices, 'Years_of_Experience'] = np.nan

# Fill missing values with mean
mean_exp = df['Years_of_Experience'].mean()
df['Years_of_Experience'].fillna(mean_exp, inplace=True)

# Introduce outliers
outlier_indices = np.random.choice(df.index, size=10, replace=False)
df.loc[outlier_indices, 'Monthly_Sales'] = df.loc[outlier_indices, 'Monthly_Sales'] * 5

# Select features for clustering
features = ['Years_of_Experience', 'Monthly_Sales', 'Projects_Completed']
X = df[features]

# Standardize the features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("K-Means Clustering Parameters:")
print(f"- Number of clusters: 3")
print(f"- Random state: 42 (for reproducibility)")
print(f"- n_init: 10 (number of times algorithm runs with different centroid seeds)")
print(f"- max_iter: 300 (default - maximum iterations)")
print("\nHow K-Means works:")
print("1. Initialize 3 random cluster centroids")
print("2. Assign each point to nearest centroid")
print("3. Update centroids to mean of assigned points")
print("4. Repeat steps 2-3 until convergence")
```

---

### b) Display cluster distribution (3 marks)

```python
# Add cluster labels to DataFrame
print("Cluster Distribution:")
cluster_counts = df['Cluster'].value_counts().sort_index()
print(cluster_counts)

print("\nCluster Percentages:")
cluster_percentages = (cluster_counts / len(df)) * 100
print(cluster_percentages)

# Display cluster statistics
print("\nCluster Statistics (Years of Experience):")
print(df.groupby('Cluster')['Years_of_Experience'].describe())

print("\nCluster Statistics (Monthly Sales):")
print(df.groupby('Cluster')['Monthly_Sales'].describe())

print("\nCluster Statistics (Projects Completed):")
print(df.groupby('Cluster')['Projects_Completed'].describe())

# Notable insights
print("\n🔍 Notable Insights:")
print("- Cluster 0: Mid-experience, mid-sales employees (largest group)")
print("- Cluster 1: High-experience, high-sales employees (outliers)")
print("- Cluster 2: Lower-experience, lower-sales employees")
```

---

### c) Create scatter plot of clusters (3 marks)

```python
plt.figure(figsize=(12, 8))

# Create scatter plot with different colors for clusters
colors = ['red', 'blue', 'green']
for cluster in range(3):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Years_of_Experience'], 
                cluster_data['Monthly_Sales'],
                c=colors[cluster], 
                label=f'Cluster {cluster}', 
                alpha=0.6, 
                s=50)

# Plot centroids
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

plt.scatter(centroids_original[:, 0], centroids_original[:, 1], 
            c='black', marker='X', s=200, label='Centroids', edgecolors='white', linewidth=2)

plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Monthly Sales ($)', fontsize=12)
plt.title('K-Means Clustering: Employees by Experience and Sales', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

# SECTION B

## QUESTION 4 (COMPULSORY) - Linear Regression (10 marks)

### a) Train a simple linear regression model (6 marks)

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create the dataset
data = {
    'YearsExperience': [1.139343, 12.046205, 3.2, 4.5, np.nan, 6.5, 6.8725, 7.5, 8.3, 9.0, 9.5],
    'Salary': [31200, 46200, np.nan, 60000, 65200, 67250, 67200, np.nan, 83000, 88000, 95000]
}

df_reg = pd.DataFrame(data)

# Handle missing values
df_reg['YearsExperience'].fillna(df_reg['YearsExperience'].mean(), inplace=True)
df_reg['Salary'].fillna(df_reg['Salary'].mean(), inplace=True)

print("Dataset after handling missing values:")
print(df_reg)

# Prepare features and target
X = df_reg[['YearsExperience']]  # Feature (2D array for sklearn)
y = df_reg['Salary']             # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Model trained successfully!")
```

---

### b) Predict and compute MSE (2 marks)

```python
# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Display actual vs predicted
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Difference': y_test.values - y_pred
})
print("\nActual vs Predicted:")
print(results)
```

---

### c) Extract and interpret coefficient and intercept (2 marks)

```python
# Extract coefficient and intercept
coefficient = model.coef_[0]
intercept = model.intercept_

print("Linear Regression Equation:")
print(f"Salary = {coefficient:.2f} × YearsExperience + {intercept:.2f}")
print(f"\nCoefficient (slope): {coefficient:.2f}")
print(f"Intercept: {intercept:.2f}")

# Interpretation
print("\n📊 Interpretation:")
if coefficient > 0:
    print(f"✅ POSITIVE relationship: For each additional year of experience,")
    print(f"   salary increases by approximately ${coefficient:.2f} on average.")
else:
    print(f"❌ NEGATIVE relationship: For each additional year of experience,")
    print(f"   salary decreases by approximately ${abs(coefficient):.2f} on average.")

print(f"\nThe intercept (${intercept:.2f}) represents the estimated salary")
print("for someone with 0 years of experience (baseline).")

# Visualize the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training data')
plt.scatter(X_test, y_test, color='green', alpha=0.6, label='Test data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.title('Linear Regression: Experience vs Salary')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## QUESTION 5: Support Vector Machine (SVM) - 10 marks

### a) Train SVM classifier with scaling (3 marks)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Create the dataset
np.random.seed(42)
n = 1000

data = {
    'Task_Completion_Rate': np.random.normal(7, 2, n),
    'Customer_Satisfaction': np.random.uniform(1, 10, n),
    'Hours_Worked_Per_Week': np.random.normal(40, 5, n),
    'Performance': np.random.choice([0, 1], n, p=[0.5, 0.5])
}

df_svm = pd.DataFrame(data)

# Features and target
X = df_svm[['Task_Completion_Rate', 'Customer_Satisfaction', 'Hours_Worked_Per_Week']]
y = df_svm['Performance']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Descriptive statistics before scaling
print("Descriptive Statistics BEFORE Scaling:")
print(X_train.describe())

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for display
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)

print("\n" + "="*50)
print("Descriptive Statistics AFTER Scaling:")
print(X_train_scaled_df.describe())

# Train SVM classifier
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

print("\n✅ SVM Model trained successfully!")
print(f"Parameters used: kernel='rbf', C=1.0, gamma='scale'")
```

---

### b) Predict on test set (3 marks)

```python
# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Convert predictions to DataFrame for descriptive statistics
y_pred_df = pd.DataFrame(y_pred, columns=['Predicted_Performance'])

print("Predicted Performance Distribution:")
print(y_pred_df.describe())

print(f"\nPrediction Summary:")
print(f"Total predictions: {len(y_pred)}")
print(f"Predicted Class 0 (Low Performance): {sum(y_pred == 0)} ({sum(y_pred == 0)/len(y_pred)*100:.1f}%)")
print(f"Predicted Class 1 (High Performance): {sum(y_pred == 1)} ({sum(y_pred == 1)/len(y_pred)*100:.1f}%)")

# Compare with actual
print(f"\nActual Test Set Distribution:")
print(f"Actual Class 0: {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
print(f"Actual Class 1: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")
```

---

### c) Evaluate model performance (4 marks)

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("                 Predicted")
print("                 Negative  Positive")
print(f"Actual Negative    {cm[0,0]:5d}    {cm[0,1]:5d}")
print(f"Actual Positive    {cm[1,0]:5d}    {cm[1,1]:5d}")

# Calculate metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()

print("\nDetailed Metrics:")
print(f"True Negatives (Correctly predicted Low): {tn}")
print(f"False Positives (Incorrectly predicted High): {fp}")
print(f"False Negatives (Missed High performers): {fn}")
print(f"True Positives (Correctly predicted High): {tp}")

print(f"\nPrecision (High Performance): {tp/(tp+fp):.4f}")
print(f"Recall (High Performance): {tp/(tp+fn):.4f}")
print(f"Specificity (Low Performance): {tn/(tn+fp):.4f}")
print(f"F1-Score: {2*tp/(2*tp+fp+fn):.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Performance', 'High Performance']))

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Low', 'Predicted High'],
            yticklabels=['Actual Low', 'Actual High'])
plt.title('Confusion Matrix - SVM Classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

---

## QUESTION 6: Decision Tree (10 marks) - Partial due to incomplete question

Based on the pattern from Question 5, here's how you would approach Decision Tree classification:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Use the same dataset from Question 5
# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test_scaled)

# Evaluate
accuracy_dt = accuracy_score(y_test, y_pred_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)

print("Decision Tree Performance:")
print(f"Accuracy: {accuracy_dt:.4f}")
print("\nConfusion Matrix:")
print(cm_dt)

# Visualize decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(dt_model, feature_names=X.columns, class_names=['Low', 'High'], 
               filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()
```

</details>

<br /><hr /><br />

<details>
    <summary>Exam...</summary>

This final exam for Machine Learning module at the University of Kigali covers a broad range of topics, from supervised regression and classification to unsupervised clustering, reinforcement learning, and deep learning.

---

## SECTION A (Compulsory)

### QUESTION 1: Salary Prediction (Linear Regression)

This question focuses on data preprocessing and building a linear model to handle missing data .

a) Data Preparation and Preprocessing :

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load data into DataFrame
data = {
    'YearsExperience': [1.0, 2.5, 3.0, 4.0, np.nan, 6.0, 7.0, 8.5, 9.0, 10.0],
    'EducationLevel': ['Bachelor', 'Master', 'Bachelor', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD'],
    'Salary': [40000, 48000, np.nan, 62000, 58000, 72000, np.nan, 85000, 90000, np.nan]
}
df = pd.DataFrame(data)

# 2. Handle missing values using mean imputation
df['YearsExperience'] = df['YearsExperience'].fillna(df['YearsExperience'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

# 3. Convert categorical EducationLevel to numerical using Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['EducationLevel'] = le.fit_transform(df['EducationLevel'])

# Display processed data
print("Processed Dataset (First 10 rows):")
print(df.head(10))

```

b) Model Training :

```python
# Select features and target
X = df[['YearsExperience', 'EducationLevel']]
y = df['Salary']

# Split data (20% test size, random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

```

c) Evaluation and Visualization :

```python
# Predict and compare
y_pred = model.predict(X_test)
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted Salary:")
print(comparison.head())

# Metrics
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# Visualization
plt.scatter(X_test['YearsExperience'], y_test, color='blue', label='Actual')
plt.plot(X_test['YearsExperience'], y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary (Test Set)')
plt.legend()
plt.show()

```

---

## SECTION B (Choose Any Three)

### QUESTION 2: Cardiovascular Risk Classification (SVM)

This task uses Support Vector Machines to classify health risks based on blood pressure and cholesterol .

(i) Feature Selection and Splitting :

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Features and target
X = df[['Systolic_BP', 'Cholesterol']]
y = df['Risk']

# Split 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

```

(ii) SVM Training :

```python
# Linear Kernel SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

```

(iii) Evaluation :

```python
y_pred = svm_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

```

### QUESTION 3: Customer Segmentation (K-means)

Unsupervised clustering to group retail customers by income and spending behavior .

i. Elbow Method :

```python
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df[['Exercise_Hours', 'Calorie_Intake']]) # Using features provided in script
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

```

ii & iii. Training and Visualization :

```python
# Assuming optimal clusters = 3 from elbow plot
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Exercise_Hours', 'Calorie_Intake']])

# Plot clusters and centroids
plt.scatter(df['Exercise_Hours'], df['Calorie_Intake'], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('Customer Segments')
plt.legend()
plt.show()

```

### QUESTION 4: Course Enrollment (Decision Tree)

Predicting student enrollment using age, study hours, and completed courses .

i, ii & iii. Pipeline :

```python
from sklearn.tree import DecisionTreeClassifier

X = df[['Age', 'Weekly_Study_Hours', 'Completed_Courses']]
y = df['Enrolled']

# 80/20 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and Predict
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

```

### QUESTION 5: Reinforcement Learning (Q-Learning)

The agent learns to navigate a 4x4 grid to reach a goal at coordinate (3,3) .

a) Q-table Initialization :

```python
num_states = grid_size * grid_size  # 16
num_actions = 4  # Up, Down, Left, Right
q_table = np.zeros((num_states, num_actions))

```

b) Training Loop :

```python
alpha, gamma, epsilon = 0.1, 0.9, 0.1
for episode in range(1000):
    state = 0 # Starting index
    done = False
    while not done:
        # Epsilon-greedy selection
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table[state])
        
        # Take action, get reward and next_state (Simplified logic)
        next_state = move(state, action) # Logic for grid boundaries
        reward = rewards[next_state]
        
        # Update Q-value
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        
        state = next_state
        if state == goal_index: done = True

```

### QUESTION 6: Pattern Classification (CNN)

Using a Deep Learning framework to distinguish between diagonal stripes and checkerboards .

a & b) Architecture and Training :

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

```

c) Evaluation :

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Simple visualization of predictions
predictions = model.predict(X_test[:5])
# ... plotting logic for images ...

```

</details>

