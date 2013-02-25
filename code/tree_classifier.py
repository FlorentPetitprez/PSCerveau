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

"""
image_shape=(150, 100)
displays=[characters.signatures_to_letter(y_test_predict[i], image_shape, .1) for i range(y_test_predict.shape)]
pl.figure()
for display in displays
	pl.imshow(display, interpolation="nearest")
	pl.gray()
	pl.show()
end for

#n_digit = 10


#pl.figure(figsize=(2. * n_digits, 2.26 * 2))
#pl.suptitle("digits completion with extra trees classifier", size=16)

#for i in xrange(1, 1 + n_digit):
 #   digit_id = np.random.randint(X_test.shape[0])

  #  true_digit = np.hstack((X_test[digit_id], Y_test[digit_id]))
   # completed_digit = np.hstack((X_test[digit_id], Y_test_predict[digit_id]))

    #pl.subplot(2, n_digit, i)
    #pl.axis("off")
    #pl.imshow(true_digit.reshape(image_shape),
     #         cmap=pl.cm.gray,
      #        interpolation="nearest")

    #pl.subplot(2, n_digit, n_digit + i)
    #pl.axis("off")
    #pl.imshow(completed_face.reshape(image_shape),
     #         cmap=pl.cm.gray,
      #        interpolation="nearest")

#pl.show()
"""
