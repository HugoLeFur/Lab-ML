from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
import seaborn as sn

print("Load IRIS dataset, check its contents:\n")
iris=load_iris()
iris.feature_names
print(iris.feature_names)
print(iris.data[0:5,:])
print(iris.target[:])
#print(iris.data)

print("Split data into training and testing parts:\n")
X=iris.data[:, :2]
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)

print("Use a Support Vector Machine for classification:\n")
SVMmodel=SVC(kernel='linear')
SVMmodel.fit(X_train,y_train)
SVMmodel.get_params()
score = SVMmodel.score(X_test,y_test)
print("\nScore :", score)

print("1.0 Let's explore more now.:\n")

X2D = iris.data[iris.target!=2, :2]
Y2D = iris.target[iris.target!=2]

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.figure(2, figsize=(8, 6))
plt.clf()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

supvectors=SVMmodel.support_vectors_
W=SVMmodel.coef_
b=SVMmodel.intercept_
import numpy as np
xgr=np.linspace(min(X[:,0]),max(X[:,0]),100)
print("\nW[:,0]",W[:,0])
print("\nW[:,1]",(W[:,1]))
print("W:",W)
print("\nb",b)
print("\nxgr",xgr)
'''ygr=-W[:,0]/W[:,1]*xgr-b/W[:,1]
plt.scatter(xgr,ygr)
plt.show()'''


from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from numpy import quantile, where, random

random.seed(11)
x, _ = make_blobs(n_samples=300, centers=1, cluster_std=.3, center_box=(4, 4))

plt.scatter(x[:,0], x[:,1])
plt.show()


SVMmodelOne = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)


SVMmodelOne.fit(x)
pred = SVMmodelOne.predict(x)
anom_index = where(pred==-1)
values = x[anom_index]

plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.axis('equal')
plt.show()

supvectorsOne=SVMmodelOne.support_vectors_

plt.scatter(X_train[:,0], X_train[:,1])
plt.scatter(supvectorsOne[:,0], supvectorsOne[:,1], color='red')
plt.title('Linearly separable data with support vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


scores = SVMmodelOne.score_samples(x)

thresh = quantile(scores, 0.01)
print(thresh)
index = where(scores<=thresh)
values = x[index]

plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.axis('equal')
plt.show()