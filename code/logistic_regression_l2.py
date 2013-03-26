import characters
import numpy as np
import pylab as pl

from sklearn.datasets import load_digits
digits=load_digits()
data=digits['data']
target=digits['target']

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier




# first test : classificator on all data, predicts the number

C = 1.0

logregl2 = LogisticRegression(penalty='l2')

train_data = data[:1000]
train_target= target[:1000]

test_data = data[1000:]
test_target= target[1000:]

logregl2.fit(train_data, train_target)

prediction = logregl2.predict(test_data)

# cross-validation for this first test

from sklearn.cross_validation import cross_val_score

score = cross_val_score(logregl2, data, target, cv=5)

clfs = [OneVsRestClassifier(LogisticRegression(penalty='l2', C=alpha)) for alpha in np.logspace(-5, 1, 10)]

scores = [cross_val_score(clf, data, target, cv=5) for clf in clfs]

scoresAlpha = np.array(scores).mean(axis=1)



# second test : classificator on separated bars, predicts the presence of the bar

y=np.array([characters.char_to_signatures(str(t))[0] for t in target])

train_data = data[:1000]
train_target= y[:1000]

test_data = data[1000:]
test_target= y[1000:]

prediction_bars = []
for i in range(16):
	logregl2.fit(train_data, train_target[:, i])
	prediction_bars.append(logregl2.predict(test_data))

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

