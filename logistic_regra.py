from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()
#print(list(iris.keys()))
#print(iris['data'])
#print(iris['target'])
#print(iris['DESCR'])
x = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)
#tain classifier
clf = LogisticRegression()
clf.fit(x,y)
example = clf.predict(([[1.6]]))
print(example)
x_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba((x_new))
plt.plot(x_new, y_prob[:,1],"g-", label="virginica")
plt.show()
#print(y)
