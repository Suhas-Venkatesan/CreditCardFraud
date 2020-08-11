import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import resample

data = pd.read_csv('creditcard.csv')

# Separate input features and class (outcome)
y = data.Class
X = data.drop('Class', axis=1)

# The data must be split into training and testing sets before resampling to avoid the chance that
# resampling causes duplicated data points in the training sets which will skew the precision
# and accuracy of the algorithm.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

# combine the training data into one dataframe.
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
not_fraud = X[X.Class == 0]
fraud = X[X.Class == 1]

# upsample minority (essentially choosing random minority instances to duplicate)
fraud_upsampled = resample(fraud, replace=True, n_samples=len(not_fraud), random_state=27)

# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

# Display new class counts
print(upsampled.Class.value_counts())

# There are now an equal number of instances of fraud and no fraud in the training set
# 1    213245
# 0    213245
# Name: Class, dtype: int64

# Next, I will undersample the majority class. This is not as effective because it simply
# randomly removes instances of the majority class, which will remove important information
# to fit the model.

# under sample majority
not_fraud_downsampled = resample(not_fraud, replace=False, n_samples=len(fraud), random_state=27)
downsampled = pd.concat([not_fraud_downsampled, fraud])

print(downsampled.Class.value_counts())

# 1    360
# 0    360
# Name: Class, dtype: int64

"""A third method I will be trying is SMOTE, or synthetic minority oversampling technique. 
It essentially synthesizes new instances of the minority class rather than duplicating instances
like the upsampling technique. The algorithm is below:

1. A random instance of the minority class is chosen
2. K of the nearest neighbors of that instance are found (usually k=5)
3. A randomly selected neighbor is chosen, and a synthetic instance is created
at a randomly selected point between the original instance and the neighbor in the feature space

This is much more effective than simple oversampling because it generates plausible examples
which are close enough to already existing examples. 
"""

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 27)
X_train, y_train = sm.fit_sample(X_train, y_train)

print("After OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train == 0)))


"""
After OverSampling, counts of label '1': 213245
After OverSampling, counts of label '0': 213245
"""


