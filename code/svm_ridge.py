Ridge 
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from sklearn.linear_model import Ridge
>>> np.random.seed(0)
>>> mean, cov, n = [1, 5], [[1,1],[1,2]], 200
>>> d = np.random.multivariate_normal(mean, cov, n)
>>> x, y = d[:, 0].reshape(-1, 1), d[:, 1]
>>> x.shape
(200, 1)
>>> ridge = mlpy.Ridge()
>>> //ridge.learn(x, y)
>>> xx = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)
>>> yy = ridge.predict(xx)
>>> fig = plt.figure(1) # plot
>>> plot = plt.plot(x, y, 'o', xx, yy, '--k')
>>> plt.show()
SVM

////////////////////////////////////SVM simple
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from sklearn.svm import SVC
>>> x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
>>> y = np.array([1, 1, 2, 2])
>>> clf = SVC()
>>> clf.fit(x, y) 
>>> xmin, xmax = x[:,0].min()-0.1, x[:,0].max()+0.1
>>> ymin, ymax = x[:,1].min()-0.1, x[:,1].max()+0.1
>>> xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
>>> xnew = np.c_[xx.ravel(), yy.ravel()]
>>> ynew = clf.predict(xnew).reshape(xx.shape)
>>> fig = plt.figure(1)
>>> plt.set_cmap(plt.cm.Paired)
>>> plt.pcolormesh(xx, yy, ynew)
>>> plt.scatter(x[:,0], x[:,1], c=y)
>>> plt.show()
////////////////////////////////////////////SVM avec données
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from sklearn.svm import SVC
f = np.loadtxt("/home/soukaina/Bureau/project_X/data/stimuli/lexique_nletters_4_block_1.txt")
>>> x, y = f[:, :2], f[:, 2]
>>> svm = SVC()
>>> xmin, xmax = x[:,0].min()-0.1, x[:,0].max()+0.1
>>> ymin, ymax = x[:,1].min()-0.1, x[:,1].max()+0.1
>>> xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
>>> xnew = np.c_[xx.ravel(), yy.ravel()]
>>> ynew = svm.predict(xnew).reshape(xx.shape)
>>> fig = plt.figure(1)
>>> plt.set_cmap(plt.cm.Paired)
>>> plt.pcolormesh(xx, yy, ynew)
>>> plt.scatter(x[:,0], x[:,1], c=y)
>>> plt.show()
