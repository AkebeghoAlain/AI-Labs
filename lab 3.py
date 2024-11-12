# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Data Collection
# Load the Titanic dataset
data = sns.load_dataset('titanic')

# Step 2: Data Cleaning
# Check for missing values
print("Missing values per column:\n", data.isnull().sum())

# Handle missing values
data['age'].fillna(data['age'].mean(), inplace=True)        # Impute missing age with mean
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)  # Impute missing embarked with mode
data.drop(['deck', 'embark_town', 'alive'], axis=1, inplace=True)  # Drop columns with too many missing values


# Step 3: Handling Outliers
# Visualize outliers for 'fare'
sns.boxplot(data['fare'])
plt.show()

# Remove outliers in 'fare'
q_low = data["fare"].quantile(0.01)
q_hi  = data["fare"].quantile(0.99)
data = data[(data["fare"] > q_low) & (data["fare"] < q_hi)]

# Step 4: Data Normalization
# Apply Min-Max scaling to 'age' and 'fare'
scaler = MinMaxScaler()
data[['age', 'fare']] = scaler.fit_transform(data[['age', 'fare']])

# Step 5: Feature Engineering
# Create new features: 'family_size' and 'title'
data['family_size'] = data['sibsp'] + data['parch']
data['title'] = data['name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Drop columns that won’t be useful for the model
data.drop(['name', 'sibsp', 'parch'], axis=1, inplace=True)

# Encode categorical variables (e.g., 'sex', 'embarked', 'class')
data = pd.get_dummies(data, columns=['sex', 'embarked', 'class', 'who', 'adult_male', 'title'], drop_first=True)

# Step 6: Feature Selection
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Feature importance with RandomForest
# Split data for feature importance analysis
X = data.drop(['survived'], axis=1)
y = data['survived']

# Model for feature importance 
feature_model = RandomForestClassifier()
feature_model.fit(X, y)
importances = feature_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("Feature importances:\n", feature_importance_df)

# Step 7: Model Building
# Split the data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train a model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
