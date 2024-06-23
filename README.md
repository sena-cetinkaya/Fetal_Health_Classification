# FETAL HEALTH CLASSIFICATION 
This repo includes machine learning models to classify fetal health to prevent child and maternal deaths. The dataset used in this repo was taken from Kaggle: [Link to the dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)

To review my work on this dataset on Kaggle; [https://www.kaggle.com/code/senacetinkaya/fetal-health-classification](https://www.kaggle.com/code/senacetinkaya/fetal-health-classification)

-----------------------------------------------------------
# DATA VISUALIZATION
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

This repo contains a histogram chart to understand the relationships between the data and a countplot to show what the values ​​are in the "fetal_health" column.

-------------------------------------
# BUILDING MACHINE LEARNING MODELS
It includes Logistic Regression, Decision Tree Classifier, K Neighbors Classifier, Random Forest Classifier, Gradient Boosting Classifier Machine Learning models applied on this repo dataset. Also included are comparisons of the success of these models.

----------------------------------------------
### Libraries Used in the Project
```
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
```
