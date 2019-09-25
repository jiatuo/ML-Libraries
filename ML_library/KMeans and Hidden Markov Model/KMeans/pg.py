import numpy as np

centers = [1]
n = 6
n_cluster = 3
x = np.array([[1,2,3,4],
            [1,4,5,6],
            [3,2,5,6],
            [2,4,5,6],
            [12,4,5,7],
            [2,51,2,5]])

centers=[]
centers.append(1)
total=[]
for i in range(1,n_cluster):
    index = centers[i-1]
    distance=np.sum((x-x[index])**2,axis=1)
    print("distance: \n", distance)
    if total==[]:
        total=distance.reshape((1,n))
    else:
        distance=distance.reshape((1,n))
        total=np.concatenate((total,distance), axis=0)
    #total.append(distance)


    print("total = \n", total)    
    near_cluster=np.min(total,axis=0)
    print("near clusters: \n", near_cluster)

    #new_cluster=near_cluster.argmax()
    new_cluster=np.argmax(near_cluster)
    print("new cluster: ",new_cluster)

    centers.append(new_cluster)
print("centers: \n", centers)

max_iter = 5
i = 0
centroids = x[centers]
while(i < max_iter):
    #distance = np.sum((x-np.expand_dims(centroids,axis=1))**2,axis=2)
    distance = np.sum((np.expand_dims(x,axis=0)-np.expand_dims(centroids,axis=1))**2,axis=2)
    print(np.expand_dims(centroids,axis=1).shape)
    print(np.expand_dims(x, axis = 0).shape)
    print((np.expand_dims(x,axis=0)-np.expand_dims(centroids,axis=1)).shape)
    print("x-np:\n",x-np.expand_dims(centroids,axis=1))
    print("distance: \n", distance)
    y = np.argmin(distance, axis=0)
    print("y = ", y)
    i+=1





b = np.arange(10)
print(np.where(b < 5))


y = np.array([0,0,0,2,5,6,6,6,6,7,8])
unique_y = np.array([0,2,5,6,7,8])
print(y == unique_y[0])
