# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing for preprocess
from sklearn.model_selection import train_test_split

# Importing the models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Importing metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# Read the dataset
data = pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')

# DATA ANALYSIS
# Display the first few rows of the data.
print(data.head())

# Learning the size of the dataset
print(data.shape)

# Learning columns
print(data.columns)

# Check for missing values.
print(data.isnull().sum())

# Summary statistics.
print(data.info())

# Categorical data in the "fetal_health" column
print(data['fetal_health'].value_counts().sum)

# DATA VISUALIZATION
data.hist(figsize=(10, 10),bins = 50)

plt.figure(figsize=(4, 3))
sns.countplot(x='fetal_health', data=data)
plt.xlabel("fetal health")
plt.ylabel("count")
plt.title('Fetal Health')
plt.show()

# BUILDING MACHINE LEARNING MODELS
# Separating features and target variable
x = data.drop('fetal_health', axis=1)
y = data['fetal_health']

# Splitting the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

# Logistic regression
# Train the model
lr = LogisticRegression (random_state=0, max_iter=1000)
lr.fit(x_train, y_train)

# Make predictions 
y_pred1 = lr.predict(x_test)

print("Confusion Matrix \n",confusion_matrix(y_test, y_pred1))

lr_score = round(lr.score(x_test,y_test)*100,2)
print("Accuracy", lr_score)

# Decision Tree
# Train the model
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(x_train, y_train)

# Make predictions 
y_pred2 = dtc.predict(x_test)

print("Confusion Matrix \n",confusion_matrix(y_test, y_pred1))

dtc_score = round(dtc.score(x_test,y_test)*100,2)
print("Accuracy", dtc_score)

# K-Nearest Neighbours 
# Train the model
k = 3
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(x_train, y_train)

# Make predictions 
y_pred5 = knn.predict(x_test)

print("Confusion Matrix \n",confusion_matrix(y_test, y_pred1))

knn_score = round(knn.score(x_test,y_test)*100,2)
print("Accuracy", knn_score)

# Random Forest
# Train model
rf = RandomForestClassifier(random_state=0)
rf.fit(x_train, y_train)

# Make predictions 
y_pred3 = rf.predict(x_test)

print("Confusion Matrix \n",confusion_matrix(y_test, y_pred1))

rf_score = round(rf.score(x_test,y_test)*100,2)
print("Accuracy", rf_score)

# Gradient Boosting
# Train the model
gbc = GradientBoostingClassifier (random_state=1)
gbc.fit(x_train, y_train)

# Make predictions 
y_pred4 = gbc.predict(x_test)

print("Confusion Matrix \n",confusion_matrix(y_test, y_pred1))

gbc_score = round(gbc.score(x_test,y_test)*100,2)
print("Accuracy", gbc_score)

# Comparing Models
model_names = ['lr', 'dtc', 'knn', 'rf', 'gbc']
accuracies = [lr_score, dtc_score, knn_score, rf_score, gbc_score]

sns.barplot(x=model_names, y=accuracies, data=data, color="lightblue")
plt.title("Model Names - Accuracies")
plt.xlabel("Model Names")
plt.ylabel("Accuracies")
plt.show()
