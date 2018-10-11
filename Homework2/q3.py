#Kaggle Titanic
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X_data = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")
X_valid = X_data.sample(frac=0.2,random_state=200)
X_train = X_data.drop(X_valid.index)
Y_data = X_data["Survived"]
Y_valid = X_valid["Survived"]
Y_train = X_train["Survived"]
ID_test = X_test["PassengerId"]

from IPython.display import display
display(X_data.head())
display(X_data.describe())
display(X_test.head())
display(X_test.describe())



def preprocess(df):
	df.drop(["Survived"],axis=1,inplace=True,errors="ignore")
	df.drop(["PassengerId","Name","Ticket","Cabin"],axis=1,inplace=True)

	print(df["Fare"].mean())
	df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)
	df["Fare"].fillna(df["Fare"].mean(),inplace=True)
	df["Age"].fillna( df["Age"].median(),inplace=True)

	df = df.join(pd.get_dummies(df["Embarked"]))
	df.drop(["Embarked"],axis=1,inplace=True)
	df = df.join(pd.get_dummies(df["Sex"]))
	df.drop(["Sex"],axis=1,inplace=True)
	df = df.join(pd.get_dummies(df["Pclass"]))
	df.drop(["Pclass"],axis=1,inplace=True)

	df["Family"] = df.apply(lambda row: 1
	    if row["SibSp"] != 0 or row["Parch"] != 0
	    else 0, axis=1)

	df["Child"] = df.apply(lambda row: 1
	    if row["Age"] < 16
	    else 0, axis=1)
	return df

print(df.to_string())

X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
X_data = preprocess(X_data)
X_test = preprocess(X_test)
display(X_train.head())

model = LogisticRegression(fit_intercept=False)
model.fit(X_train,Y_train)

Y_pred = model.predict(X_valid)
print("Accuracy is {acc}".format(acc = accuracy_score(Y_valid, Y_pred)))
#Accuracy is 0.7865168539325843
#part c
partc = LogisticRegression(fit_intercept=False)
partc.fit(X_data, Y_data)

print("Coefficient = {coeff}".format(coeff=partc.coef_))
# Coefficient = [[-0.02154119 -0.59903285 -0.31918963  0.00320263  0.27199     0.37586675
#   -0.01290163  1.61650562 -0.9815505   1.08778194  0.31062274 -0.76344957
#    0.69802998  1.2110554 ]]

#part d
Y_test = partc.predict(X_test)
ans = pd.DataFrame({"PassengerId":ID_test,"Survived":Y_test})
ans.to_csv("submit.csv", index=False)
