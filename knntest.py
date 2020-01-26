import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

data = pd.read_csv("heart.csv")

# print(data.head())
# print(data.target.value_counts())
# print(data.values)
# sns.countplot(x="sex", data=data, palette="bwr")
# plt.show()

# plt.scatter(x=data.age[data.target==1], y=data.thalach[(data.target==1)], c="green")
# plt.scatter(x=data.age[data.target==0], y=data.thalach[(data.target==0)], c = 'black')
# plt.legend(["Disease", "Not Disease"])
# plt.xlabel("Age")
# plt.ylabel("Maximum Heart Rate")
# plt.show()

X = data.iloc[:,:-1].values
y = data.iloc[:,13].values
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.25, random_state= 0)


# normalize the data which can be done using StandardScaler() from sci-kit learn.
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Our next step is to K-NN model and train it with the training data. Here n_neighbors is the value of factor K.

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier = classifier.fit(X_train,y_train)

	
	
# y_pred = classifier.predict(X_test)
# #check accuracy
# accuracy = metrics.accuracy_score(y_test, y_pred)
# print('Accuracy: {:.2f}'.format(accuracy))
# accuracy 0.82

classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier = classifier.fit(X_train,y_train)
#prediction
y_pred = classifier.predict(X_test)
#check accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))	

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)








