import numpy as np
import copy


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
    
    y_l = copy.deepcopy(y)


    for i in range(N):
        if(y_l[i] == 0):
            y_l[i] = -1


    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        ############################################


        w = np.zeros(D)
        b = 0
        one_cols = np.ones((N, 1))
        X_1 = np.hstack((one_cols, X))
        B = np.array([b])
        w_1 = np.hstack((B, w))

        for i in range(max_iterations):
            w_T_X = np.dot(X_1, w_1)
            total = 0
            for j in range(N):
                if(y_l[j] * w_T_X[j] <= 0):
                    total = total + np.dot(y_l[j], X_1[j])
            gradient = total / N
            w_1 = w_1 + step_size * gradient
        b = w_1[0]
        w = w_1[1:]

        
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        ############################################
        one_cols = np.ones((N, 1))
        X_1 = np.hstack((one_cols, X))
        B = np.array([b])
        w_1 = np.hstack((B, w))

        for i in range(max_iterations):
            sig = sigmoid(np.dot(X_1, w_1) + b)
            gradient = np.dot(X_1.T, (sig - y)) / N
            w_1 = w_1 - step_size * gradient
        
        b = w_1[0]
        w = w_1[1:]

            


    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    ############################################
    value = 1 / (1 + np.exp(-z))
    return value



def softmax(z):
    z = np.exp(z - np.amax(z))
    z_sum = np.sum(z, axis = 1)

    return (z.T/z_sum).T

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################
        for i in range(N):
            result =  np.dot(X, w) + b
            if(result[i] > 0):
                preds[i] = 1
            else:
                preds[i] = 0

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################
        preds = np.round(sigmoid(np.dot(X, w) + b))

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################
        # y = np.eye(C)[y].T
        # for i in range(max_iterations):
        #     n = np.random.choice(N)


        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################

        y = np.eye(C)[y]

        for i in range(max_iterations):

            sm = softmax((np.dot(w,X.T)).T + b)
            gradient_w = np.dot((sm - y).T,X) / N
            gradient_b = np.sum((sm - y), axis = 0) / N
            w = w - gradient_w * step_size
            b = b - gradient_b * step_size

  

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    ############################################




    preds = softmax((np.dot(w,X.T)).T + b)
    preds = np.argmax(preds, axis = 1)

    assert preds.shape == (N,)
    return preds




        