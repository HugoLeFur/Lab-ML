import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist



np.random.seed(14)
number_of_point = 100 
features = 2
k = 8
data_set=np.random.random((number_of_point,features))
plt.scatter(data_set[:,0], data_set[:,1])
#plt.show()

#To do list :
#Init centroid and get distance to them
#Verif convergence
#Update clusters
#Update centers

#Init centroid
centroids = data_set[np.random.choice(number_of_point, k, replace=False)]
not_converged = True
closest = np.zeros(number_of_point)

while(not_converged):

    old_closest = closest.copy()

    distances = np.zeros((number_of_point,k))
    distances = cdist(data_set, centroids)
    print(distances)

    #Update clusters
    closest = np.argmin(distances, axis=1)

    #Update centroids
    for i in range(k):
        centroids[i,:]=data_set[closest==i].mean(axis=0)

    #Verif convergence
    if all(closest == old_closest):
        not_converged=False

    plt.scatter(data_set[:,0],data_set[:,1],c=closest)
    plt.scatter(centroids[:,0],centroids[:,1],marker="d")
    print("\n",centroids)
    plt.show()

