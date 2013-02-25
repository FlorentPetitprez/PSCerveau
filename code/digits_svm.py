import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
digits=load_digits()
data=digits['data']
target=digits['target']

from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier

svm = SVC(C=.001)

ovrest_svm = OneVsRestClassifier(svm)

train_data = data[:1000]
train_target= target[:1000]

test_data = data[1000:]
test_target= target[1000:]

ovrest_svm.fit(train_data, train_target)

prediction= ovrest_svm.predict(test_data)

from sklearn.cross_validation import cross_val_score

score = cross_val_score(ovrest_svm, data, target, cv=5)

clfs = [OneVsRestClassifier(SVC(C=alpha)) for alpha in np.logspace(-5, 1, 10)]

scores = [cross_val_score(clf, data, target, cv=5) for clf in clfs]


