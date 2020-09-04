from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from Resampling import data, upsampled, downsampled, y, outlier_fraction

"""The first unsupervised algorithm I will be trying is the isolation forest 
algorithm with the original data (cut to 10% of its original size for faster
computation). """

# Re-setting X to the data minus the Class feature
X = data.drop('Class', axis=1)

# Fitting the model
a = IsolationForest(max_samples=len(X), contamination=outlier_fraction).fit(X)

# Prediction
y_prediction = a.predict(X)

y_prediction[y_prediction == 1] = 0  # Valid transactions are labelled as 0.
y_prediction[y_prediction == -1] = 1  # Fraudulent transactions are labelled as 1.

errors = (y_prediction != y).sum()  # Total number of errors is calculated.

print(errors)
print(accuracy_score(y_prediction, y))
print(classification_report(y_prediction, y))

"""Now, let's try the same isolation forest algorithm with a randomly upsampled dataset with 
21326 fraudulent and 21326 valid transactions, where the outlier fraction
is .5"""

X = upsampled.drop('Class', axis=1)
y = upsampled.Class

# New model with 50% contamination because there are equal amounts of fraud and valid
b = IsolationForest(max_samples=len(X), contamination=.50).fit(X)

# Prediction
y_prediction2 = a.predict(X)

y_prediction2[y_prediction2 == 1] = 0  # Valid transactions are labelled as 0.
y_prediction2[y_prediction2 == -1] = 1  # Fraudulent transactions are labelled as 1.

errors2 = (y_prediction2 != y).sum()  # Total number of errors is calculated.

print(errors2)
print(accuracy_score(y_prediction2, y))
print(classification_report(y_prediction2, y))

"""The precision with the normal data set was around 37%, while the precision for 
the upsampled dataset was 40%. The recall with the normal dataset was only around 
30% (false negatives), while the recall for the upsampled dataset was much better,
at around 65%. 
This shows that up sampling the data before using isolation forest results in
better accuracy and less false negatives and false positives


Next, we will try the local outlier factor algorithm. First, we will use the original
dataset (but 1/10 of the original size)

"""

# Reset the variables for X and y for the original data
X = data.drop('Class', axis=1)
y = data.Class # y is output

# Initialize a model with 20 neighbors
c = LocalOutlierFactor(n_neighbors = 20,contamination = outlier_fraction)
# Fit the model
y_prediction3 = c._fit_predict(X)
y_prediction3[y_prediction3 == 1] = 0 # Valid transactions are labelled as 0.
y_prediction3[y_prediction3 == -1] = 1 # Fraudulent transactions are labelled as 1.

errors3 = (y_prediction3 != y).sum()
print(errors3)
print()