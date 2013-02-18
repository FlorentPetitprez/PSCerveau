import numpy as np

from sklearn.datasets import load_digits
digits=load_digits()
data=digits['data']
target=digits['target']

from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier

svm = SVC()

ovrest_svm = OneVsRestClassifier(svm)

train_data = data[:1000]
train_target= target[:1000]

test_data = data[1000:]
test_target= target[1000:]

ovrest_svm.fit(train_data, train_target)

prediction= ovrest_svm.predict(test_data)


