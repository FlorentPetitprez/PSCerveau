import numpy as np

from sklearn.datasets import load_digits
digits=load_digits()
data=digits['data']
target=digits['target']

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.multiclass import OneVsRestClassifier

logregl1 = logreg(penalty='l1')

train_data = data[:1000]
train_target= target[:1000]

test_data = data[1000:]
test_target= target[1000:]

logregl1.fit(train_data, train_target)

prediction= logregl1.predict(test_data) # premiere prediction qui permettrait de faire un affichage (sans lien avec la cross validation)

from sklearn.cross_validation import cross_val_score

score = cross_val_score(logregl1, data, target, cv=5)

clfs = [OneVsRestClassifier(logreg(penalty='l1', C=alpha)) for alpha in np.logspace(-5, 1, 10)]

scores = [cross_val_score(clf, data, target, cv=5) for clf in clfs]



