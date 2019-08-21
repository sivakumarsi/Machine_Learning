## ML program using Decision Tree for classifying gender based on attributes like height, weight,shoe size

from sklearn import tree
from sklearn.linear_model import LogisticRegression


#Using DecisionTreeClassifier model

#[height, weight, shoesize]

X = [[180,80,44] ,[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,55,37],[171,75,42],[181,85,43]]

Y = ['male','female','female','female','male','male','male','female', 'male','female','male']

clf= tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

prediction = clf.predict([[190,70,43]])

print('Prediction using Decision Tree =', prediction)

#Use LogisticRegression model

clf_lr = LogisticRegression()
clf_lr = clf_lr.fit(X,Y)

prediction_lr = clf_lr.predict([[190,70,43]])

print('Prediction using LogisticRegression =', prediction_lr)




