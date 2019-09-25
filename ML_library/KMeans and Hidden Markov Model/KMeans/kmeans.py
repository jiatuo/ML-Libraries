import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    
    #first center randomly chosen:
    first_center = generator.randint(0, n)
    centers = [first_center]
    all_distances = []
    for i in range(0, n_cluster - 1):
        distances = np.sum((x - x[centers[i]]) ** 2, axis=1)
        all_distances.append(distances)
        distances_to_nearest_cluster = np.min(all_distances, axis=0)
        new_cluster = np.argmax(distances_to_nearest_cluster)
        centers.append(new_cluster)



    

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        y = np.zeros(N)
        centroids = x[self.centers]
        #preprocessing
        #1 X N X D
        x_p = np.expand_dims(x, axis=0)
        J = 0
        i = 0
        while(i < self.max_iter):
            #C X 1 X D
            centroids_p = np.expand_dims(centroids, axis=1)
            J_new = 0
            #C X N X D
            distances = np.sum((x_p - centroids_p) ** 2, axis=2)
            #(N,)
            y = np.argmin(distances, axis=0)
            J_new = np.sum(np.min(distances, axis=0))

            #test for convergence
            if np.abs(J_new - J) < self.e:
                i += 1
                break
            
            J = J_new

            #update centroids
            unique_centroids = np.unique(y)
            for j in range(0, len(unique_centroids)):
                centroids[unique_centroids[j]] = np.mean(x[np.where(unique_centroids[j] == y)], axis=0)

            i += 1

        self.max_iter = i



        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        r = np.zeros(N)
        #get initial centers
        centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        centroids = x[centers]
        #preprocessing
        #1 X N X D
        x_p = np.expand_dims(x, axis=0)
        J = 0
        i = 0
        while(i < self.max_iter):
            #C X 1 X D
            centroids_p = np.expand_dims(centroids, axis=1)
            J_new = 0
            #C X N X D
            distances = np.sum((x_p - centroids_p) ** 2, axis=2)
            #(N,)
            r = np.argmin(distances, axis=0)
            J_new = np.sum(np.min(distances, axis=0))

            #test for convergence
            if np.abs(J_new - J) < self.e:
                i += 1
                break
            
           

            #update centroids
            unique_centroids = np.unique(r)
            for j in range(0, len(unique_centroids)):
                centroids[unique_centroids[j]] = np.mean(x[np.where(unique_centroids[j] == y)], axis=0)

            i += 1



        centroid_labels = np.zeros(self.n_cluster)
        unique_centroids = np.unique(r)
        for centroid in unique_centroids:
            labels_for_all_points_of_this_centroid = y[np.where(r == centroid)]
            points_of_this_centroid, count = np.unique(labels_for_all_points_of_this_centroid, return_counts=True)
            label = np.argmax(count)
            centroid_labels[centroid] = points_of_this_centroid[label]


        

        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        labels = np.zeros(N)
        for i in range(N):
            distances = np.sum((x[i] - self.centroids) ** 2, axis=1)
            labels[i] = self.centroid_labels[np.argmin(distances)]

        
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    x, y, RGB = image.shape
    new_im = np.zeros((x, y, RGB))
    for i in range(x):
        for j in range(y):
            distances = np.sum((image[i, j] - code_vectors) ** 2, axis=1)
            new_im[i, j] = code_vectors[np.argmin(distances)]


    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im
