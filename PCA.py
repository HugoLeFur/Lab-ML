import numpy as np
# Defined 3 points in 2D-space:
X=np.array([[2, 1, 0],[4, 3, 0]])
# Calculate the covariance matrix:
R=np.matmul(X,X.T)/3
print(R)

# Calculate the SVD decomposition and new basis vectors:
[U,D,V]=np.linalg.svd(R)  # call SVD decomposition
u1=U[:,0] # new basis vectors
u2=U[:,1]

# Calculate the coordinates in new orthonormal basis:
Xi1=np.matmul(np.transpose(X),u1)
Xi2=np.matmul(np.transpose(X),u2)
print(Xi1,Xi2)

# Calculate the approximation of the original from new basis
#print(Xi1[:,None]) # add second dimention to array and test it
Xaprox=np.matmul(u1[:,None],Xi1[None,:])+np.matmul(u2[:,None],Xi2[None,:])

# Check that you got the original
print(Xaprox)





# Load Iris dataset as in the last PC lab:
from sklearn.datasets import load_iris
iris=load_iris()
iris.feature_names
print(iris.feature_names)
print(iris.data[0:5,:])
print(iris.target[:])
     

# We have 4 dimensions of data, plot the first three colums in 3D
X=iris.data
y=iris.target
import matplotlib.pyplot as plt
axes1=plt.axes(projection='3d')
axes1.scatter3D(X[y==0,1],X[y==0,1],X[y==0,2],color='green')
axes1.scatter3D(X[y==1,1],X[y==1,1],X[y==1,2],color='blue')
axes1.scatter3D(X[y==2,1],X[y==2,1],X[y==2,2],color='magenta')
plt.show()

     

# Pre-processing is an important step, you can try either StandardScaler (zero mean, unit variance of features)
# or MinMaxScaler (to interval from 0 to 1)
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
Xscaler = StandardScaler()
Xpp=Xscaler.fit_transform(X)

# define PCA object (three components), fit and transform the data
pca = decomposition.PCA(n_components=3)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
print(pca.get_covariance())
# you can plot the transformed feature space in 3D:
axes2=plt.axes(projection='3d')
axes2.scatter3D(Xpca[y==0,0],Xpca[y==0,1],Xpca[y==0,2],color='green')
axes2.scatter3D(Xpca[y==1,0],Xpca[y==1,1],Xpca[y==1,2],color='blue')
axes2.scatter3D(Xpca[y==2,0],Xpca[y==2,1],Xpca[y==2,2],color='magenta')
plt.show()



plt.scatter(Xpca[y==0,0],Xpca[y==0,1],color='green')
plt.scatter(Xpca[y==1,0],Xpca[y==1,1],color='blue')
plt.scatter(Xpca[y==2,0],Xpca[y==2,1],color='magenta')
plt.show()




# Import train_test_split as in last PC lab, split X (original) into train and test, train KNN classifier on full 4-dimensional X

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
knn1=KNeighborsClassifier(n_neighbors = 3)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
knn1.fit(X_train,y_train)
Ypred=knn1.predict(X_test)
# Import and show confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm=confusion_matrix(y_test,Ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

knn1=KNeighborsClassifier(n_neighbors = 3)
X_trainPCA, X_testPCA, y_trainPCA, y_testPCA = train_test_split(Xpca,y,test_size=0.3)
knn1.fit(X_trainPCA,y_trainPCA)
YpredPCA=knn1.predict(X_testPCA)
# Import and show confusion matrix
cm=confusion_matrix(y_testPCA,YpredPCA)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

knn1=KNeighborsClassifier(n_neighbors = 3)
X_trainOriginal, X_testOriginal, y_trainOriginal, y_testOriginal = train_test_split(X[:,0:1],y,test_size=0.3)
knn1.fit(X_trainOriginal,y_trainOriginal)
YpredOriginal=knn1.predict(X_testOriginal)
# Import and show confusion matrix
cm=confusion_matrix(y_testOriginal,YpredOriginal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()