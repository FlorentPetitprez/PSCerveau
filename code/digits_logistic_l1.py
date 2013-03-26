import numpy as np
import pylab as pl

from sklearn.datasets import load_digits
digits=load_digits()
data=digits['data']
target=digits['target']

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.multiclass import OneVsRestClassifier

import characters


logregl1 = logreg(penalty='l1')


# cross validation
#from sklearn.cross_validation import cross_val_score

#score = cross_val_score(logregl1, data, target, cv=5)

#clfs = [OneVsRestClassifier(logreg(penalty='l1', C=alpha)) #for alpha in np.logspace(-5, 1, 10)]

#scores = [cross_val_score(clf, data, target, cv=5) for clf in clfs]

#scores_mean = np.array(scores).mean(axis=1) # calculation of the mean score for each value of alpha
# end of cross validation


# prediction and display of the result
# we try to predict the presence or not of each bar, and not a whole digit.

y=np.array([characters.char_to_signatures(str(t))[0] for t in target])

train_data = data[:1000]
train_target= y[:1000]

test_data = data[1000:]
test_target= y[1000:]

prediction_bars = []

for i in range(12):
	logregl1.fit(train_data, train_target[:, i])
	predic = logregl1.predict(test_data)
	prediction_bars.append(predic)

# the following is because fit raises an exception "The number of classes has to be greater than one." when called for i in [12,15] (diagonal bars).
# Indeed, we cannot fit because the value is always 0 for those bars.
for i in range(4):
	prediction_bars.append(np.zeros(predic.shape))

# each vector of prediction_bars contains the signature of the predicted character.
prediction_bars = np.array(prediction_bars).T


#enables the display of the set of 200 predictions, starting at position number*200. Number must be between in [0,3]
def display_prediction(number=0):
	pl.figure()
	if (number<3):
		for i in range(200):
			pl.subplot(20, 20, 2*i + 1)
			display = test_data[i+(number*200)].reshape(8,8)
			pl.imshow(display)
			pl.axis('off')
			pl.gray()		
			pl.subplot(20, 20, 2*i + 2)
			display = characters.signatures_to_letter(prediction_bars[i+(number*200)], (150,100), .1)
			pl.imshow(display, interpolation="nearest")
			pl.axis('off')
			pl.gray()
	elif (number==3):
		for i in range(197):
			pl.subplot(20, 20, 2*i + 1)
			display = test_data[i+600].reshape(8,8)
			pl.imshow(display)
			pl.axis('off')
			pl.gray()		
			pl.subplot(20, 20, 2*i + 2)
			display = characters.signatures_to_letter(prediction_bars[i+600], (150,100), .1)
			pl.imshow(display, interpolation="nearest")
			pl.axis('off')
			pl.gray()
	pl.show()
display_prediction(number=0)


