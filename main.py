import pandas # Data frames
import numpy # Arrays
from sklearn.model_selection import train_test_split # Splitting dataset into train and test sets
from sklearn.linear_model import LogisticRegression # Model training
from sklearn.metrics import accuracy_score # Checking model-performance
print("-\nWelcome to my heart disease prediction ML algo :D \n")

# Loading .csv heart data into a Pandas dataframe
data_heartCondition = pandas.read_csv("heart_disease_data.csv") # this file is included in the repository
# target = 0 --> person has heart disease? nope
# target = 1 --> person has heart disease? yeppers

# Exposition of the dataset (no modifications made)
print("GENERAL INFO about dataset-size, data-type of cell, and memory usage of dataframe") #
print(data_heartCondition.info(), "\n\n\n")
print("STATISTICAL MEASUREMENTS, incl. standard deviation and all quartile values") #
print(data_heartCondition.describe(), "\n\n")
print("Checking the counts of 0s and 1s in the Target column") # to gauge the distribution of people with heart disease
print(data_heartCondition["target"].value_counts())
print("The distribution is almost equal, meaning that additional pre-processing is not necessary.\n")

# Separating the target column from all the other features
nonTarget_features = data_heartCondition.drop(columns = "target", axis = 1) # axis=1 => dropping a column (as opposed to a row)
target_feature = data_heartCondition["target"]
print("All non-target features") #
print(nonTarget_features, "\n")
print("Target feature only") #
print(target_feature, "\n\n")

# Splitting the data into training and testing sets (test set gets 20% of the original data; train set gets 80%)
nonTarget_train, nonTarget_test, target_train, target_test = train_test_split(nonTarget_features, target_feature, test_size = 0.2, stratify = target_feature, random_state = 404)
# stratify=target_feature => distribution of target=0 and target=1 in both the training and testing sets is as same as possible to the distribution in the original dataset
# random_state=404 => using the randomization seed-number 404
print("The amount of rows/columns of pre-split vs. train vs. test (rows, columns)")
print("All non-target features:", nonTarget_features.shape, nonTarget_train.shape, nonTarget_test.shape)
print("Target feature only: ", target_feature.shape, target_train.shape, target_test.shape)

# Training the model using Logistic Regression (chosen to model binary outputs)
model_logReg = LogisticRegression()
model_logReg.fit(nonTarget_train, target_train) # feeding training data into logistic regression model

# Evaluating the accuracy of the model on TRAIN-data
nonTarget_trainPrediction = model_logReg.predict(nonTarget_train)
trainData_accuracy = accuracy_score(nonTarget_trainPrediction, target_train)
print("Model accuracy on train-data: ", trainData_accuracy) # train-fit accuracy-score ≈ 85.124 %

# Evaluating the accuracy of the model on TEST-data
nonTarget_testPrediction = model_logReg.predict(nonTarget_test)
testData_accuracy = accuracy_score(nonTarget_testPrediction, target_test)
print("Model accuracy on test-data: ", testData_accuracy, "\n") # test-fit accuracy-score ≈ 85.246 %

# Finalized prediction system (arbitrary input-data --> output value)
age = int(input("(1/13) Enter an integer indicating your patient's age: "))
sex = int(input("(2/13) Enter 0 if your patient is male; enter 1 if female: "))
cp = int(input("(3/13) Chest Pain: enter 0 for typical angina; 1 for atypical angina; 2 for non-anginal pain; 3 if asymptomatic: "))
trestbps = int(input("(4/13) Enter an integer indicating your patient's resting blood pressure in mm Hg: "))
chol = int(input("(5/13) Enter an integer indicating your patient's serum cholesterol in mg/dL: "))
fbs = int(input("(6/13) Fasting blood sugar: Enter 1 if your patient's result is above 120 mg/dL; 0 if not: "))
restecg = int(input("(7/13) Resting Electrocardiography: Enter 0 if your patient's result is normal; 1 if having ST-T wave abnormality; 2 if showing probable or definite left ventricular hypertrophy: "))
thalach = int(input("(8/13) Enter an integer indicating your patient's maximum heart rate achieved (in BPM) during a stress test: "))
exang = int(input("(9/13) Enter 1 if your patient has exercise-induced angina; 0 if not: "))
oldpeak = float(input("(10/13) Enter a number (not necessarily an integer) indicating your patient's oldpeak in mm: "))
slope = int(input("(11/13) Enter 0 if your patient's ST-segment is upsloping; 1 if flat; 2 if downsloping: "))
ca = int(input("(12/13) Enter the number of major vessels colored by fluoroscopy (0, 1, 2, 3, or 4): "))
thal = int(input("(13/13) Thalium stress test: Enter 0 if your patient's result is normal; 1 if showing fixed defect; 2 if reversible defect; 3 if not described: "))
input_nonTarget = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) # 13 arbitrary input-values in the order of how features are ordered in heart_disease_data.csv

input_nonTarget_as_numpyArray = numpy.asarray(input_nonTarget) # conversion of input row into a numPy array
input_nonTarget_reshaped = input_nonTarget_as_numpyArray.reshape(1, -1)
output_prediction = model_logReg.predict(input_nonTarget_reshaped)
print("output in list-format:", output_prediction)

# Output reformatting
if (output_prediction[0] == 0):
    print("Your indicators suggest that your patient does NOT suffer a heart disease.")
else:
    print("Your indicators suggest that your patient suffers a heart disease.")