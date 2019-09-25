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
    value = np.max(z, axis = 0)
    value = np.reshape(value, (1, value.shape[0]))
    v_max = np.subtract(z, value)
    
    v_exp = np.exp(v_max)
    v_exp_sum = np.sum(v_exp, axis = 0)
    v_exp_sum = np.reshape(v_exp_sum, (1, N))

    return np.divide(v_exp, v_exp_sum)

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