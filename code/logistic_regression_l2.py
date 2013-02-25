import numpy as np

from sklearn.datasets import load_digits
digits=load_digits()
data=digits['data']
target=digits['target']

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

C = 1.0

logregl2 = LogisticRegression(penalty='l2')

train_data = data[:1000]
train_target= target[:1000]

test_data = data[1000:]
test_target= target[1000:]

logregl2.fit(train_data, train_target)

prediction= logregl2.predict(test_data)

from sklearn.cross_validation import cross_val_score

score = cross_val_score(logregl2, data, target, cv=5)

clfs = [OneVsRestClassifier(LogisticRegression(penalty='l2', C=alpha)) for alpha in np.logspace(-5, 1, 10)]

scores = [cross_val_score(clf, data, target, cv=5) for clf in clfs]

scoresAlpha = np.array(scores).mean(axis=1)
