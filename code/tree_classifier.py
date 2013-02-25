import characters
import numpy as np
import pylab as pl

from sklearn.datasets import load_digits
digits=load_digits()

from sklearn.ensemble import ExtraTreesClassifier

data=digits['data']
tar=digits['target']
y=np.array([characters.char_to_signatures(str(t))[0] for t in tar])

train_data = data[:1000]
test_data = data[1000:]

train_tar= y[:1000]
test_tar= y[1000:]


forest = ExtraTreesClassifier()  	

forest.fit(train_data, train_tar)
y_test_predict = forest.predict(test_data)

from sklearn.cross_validation import KFold

def cross_val_score(clf, X, y, cv=None):
    if cv is None:
	cv = KFold(len(X), 5)

    scores= []
    for train, test in cv:
        clf.fit(X[train], y[train])
        score = clf.score(X[test], y[test])
        scores.append(score)

    return scores


score = cross_val_score(forest, data, y)

#affichage des chiffres
image_shape=(8, 8)
displays2=[]
import math
d=math.trunc(np.sqrt(y_test_predict.shape[0]))
for i in range(d+1):
   for j in range(d):
     displays=[]
     displays.append(15*characters.signatures_to_letter(y_test_predict[i*j], image_shape, .1))
     displays.append(digits['images'][i*j+1000])
   displays2.append(np.hstack(displays))
display=np.vstack(displays2)
pl.figure()
pl.imshow(display)
pl.gray()
pl.show()

