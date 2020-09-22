#Breast Cancer detection using Logistic Regression


#Importing the libraries
import pandas as pd 
from termcolor import colored
print()
print(colored("*********************************","green"))
print(colored("Libraries Imported Sucessfully","green"))

#Importing the dataset
dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values   #Specifying 1 to ignore sample code number given it has no effect on results
y = dataset.iloc[:, -1].values
print(colored("Dataset Imported Sucessfully","green"))


#Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(colored("Dataset Sucessfully Split","green"))

#Training the Logistic Regression model on the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
print(colored("Logistic Regression Model Trained Sucessfully","green"))

#Predicting the test set results
y_pred = classifier.predict(X_test)
print(colored("Test Set Results Predicted","green"))

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(colored("Confusion Matrix Sucessfully","green"))

#Computing the accuracy
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(colored("Accuracy Computed Sucessfully","green"))
print(colored("*********************************","green"))
print(colored("*****VALUES FOUND*****","blue"))
print(colored("Predictions","blue"))
print(y_pred)
print(colored("Confusion Matrix","blue"))
print(cm)
print(colored("Accuracy and Deviation","blue"))
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("STD Deviation: {:.2f} %".format(accuracies.std()*100))
print()