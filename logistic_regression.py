# -*- coding: utf-8 -*-

"""
The provided Python program implements logistic regression from scratch using gradient descent on a dataset with two input variables and one output variable. Here is a summary of the main components and functionalities of the program:

Sigmoid Function:
    The sigmoid(z) function calculates the sigmoid of the input z, which is used to convert any real-valued number into a probability between 0 and 1.

Cost Function:
    The cost_function(X, y, params) calculates the logistic regression cost function, which measures the error between the predicted probabilities and the actual labels.

Gradient Calculation:
    The gradient(X, y, params) function computes the gradient of the cost function with respect to the model parameters.

Gradient Descent:
    The gradient_descent(X, y, params, learning_rate, num_iterations) function iteratively updates the model parameters using gradient descent to minimize the cost function.

Data Loading and Normalization:
    The load_and_normalize_data(file_path) function loads the training and test data from CSV files (ds1_train.csv and ds1_test.csv).
    The input features (X) are normalized using z-score normalization.

Decision Boundary Plot:
    The training data is plotted on a 2D plot, with positive and negative examples represented by different markers.
    The decision boundary, obtained using the optimized model parameters, is plotted to separate the two classes.

Model Evaluation:
    The logistic regression model's accuracy is evaluated on the test data.

The logistic regression model learns to classify data points into two classes (0 or 1) based on the given training data. 
The decision boundary represents the line that best separates the two classes. 
The accuracy on the test data indicates how well the model generalizes to unseen data.
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression

# load dataset
def read_csv_file(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y

train_file_path = 'ds1_train.csv'
test_file_path = 'ds1_test.csv'
X_train, y_train = read_csv_file(train_file_path)
X_test, y_test = read_csv_file(test_file_path)

print("First five elements in X_train before normalisation are:\n", X_train[:5])
print("Type of X_train:",type(X_train))
print ('The shape of X_train is: ' + str(X_train.shape))
print ('We have m = %d training examples' % (len(y_train)))

# Plot the training data before scaling
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label="Class 0", marker="o")
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label="Class 1", marker="x")
plt.xlabel("Input Variable 1")
plt.ylabel("Input Variable 2")
plt.title("Training Data before scaling")
plt.legend()
plt.show()

def robust_scaling(X):
    
    """
    Perform robust scaling on the input matrix X.

    Parameters:
        X (numpy array): Input feature matrix of shape (num_examples, num_features).
        
    Returns:
        X_scaled (numpy array): Normalized feature matrix
        based on their median and interquartile range, making it more robust to outliers.
    """
    median_vals = np.median(X, axis=0)
    quartile_1 = np.percentile(X, 25, axis=0)
    quartile_3 = np.percentile(X, 75, axis=0)
    IQR = quartile_3 - quartile_1
    X_scaled = (X - median_vals) / IQR
    return X_scaled

X_train=robust_scaling(X_train)
X_test=robust_scaling(X_test)

plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label="Class 0", marker="o")
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label="Class 1", marker="x")
plt.xlabel("Input Variable 1")
plt.ylabel("Input Variable 2")
plt.title("Training Data after scaling")
plt.legend()
plt.show()

print("First five elements in x1_train after normalisation are:\n", X_train[:5])

def sigmoid(z):
    """
    Computes the sigmoid of z
    Argument passed:
        z (ndarray): A scalar, numpy array of any size.
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    g=1/(1+np.exp(-z))
    return g

def compute_cost(x, y, w, b, *argv):
    """
    Computes the cost (binary cross-entropy) over all examples
    Arguments passed:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost 
    """
    m, n = x.shape
    total_cost=0
    for i in range (m):
        z_i=np.dot(x[i],w)+b
        f_wb_i=sigmoid(z_i)
        tc =  -y[i]*math.log(f_wb_i) - (1-y[i])*math.log(1-f_wb_i)
        total_cost+=tc
    total_cost=total_cost/m
    return total_cost

m, n = X_train.shape

# Compute and display cost with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w and b (zeros): {:.3f}'.format(cost))

def compute_gradient(X, y, w, b, *argv): 
    """
    Computes the gradient for logistic regression 
 
    Arguments passed:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          
        err_i  = f_wb_i  - y[i]                       
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   
    dj_db = dj_db/m 
    return dj_db, dj_dw

# Compute and display gradient with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w and b (zeros):{dj_db}' )
print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}' )

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X :    (ndarray Shape (m, n) data, m examples by n features
      y :    (ndarray Shape (m,))  target value 
      w_in : (ndarray Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)              Initial value of parameter of the model
      cost_function :              function to compute cost
      gradient_function :          function to compute gradient
      alpha : (float)              Learning rate
      num_iters : (int)            number of iterations to run gradient descent
      lambda_ : (scalar, float)    regularization constant
      
    Returns:
      w : (ndarray Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.005

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)

# Calculate the intercept and gradient of the decision boundary.
c = -b/w[1]
m = -w[0]/w[1]

# Plot the data and the classification with the decision boundary.
xmin, xmax = -2, 2
ymin, ymax = -1, 10
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], marker='o', label='Negative')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], marker='x', label='Positive')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.legend()
plt.title('Data and Decision Boundary')
plt.show()

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w and b
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    m, n = X.shape   
    p = np.zeros(m)
    for i in range(m):   
            z_wb = 0
            for j in range(n): 
                z_wb_ij = X[i, j] * w[j]
                z_wb += z_wb_ij
            z_wb += b
            f_wb = sigmoid(z_wb)
            p[i] = f_wb >= 0.5
    return p

p = predict(X_test, w,b)
print("First five elements in y_test are:\n", y_test[:10])
print("Type of y_test:",type(y_test))
print ('The shape of y_test is: ' + str(y_test.shape))

def scikit_learn(x_train,y_train,x_test):
    lr_model = LogisticRegression()
    lr_model.fit(x_train, y_train)
    y_test = lr_model.predict(x_test)
    print("Prediction on training set using scikit-learn:", y_test[:10])
    print("Accuracy on training set using scikit-learn:", lr_model.score(x_test, y_test))
    plt.scatter(x_test[y_test == 0][:, 0], x_test[y_test == 0][:, 1], marker='o', label='Negative')
    plt.scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], marker='x', label='Positive')
    
    x1_min, x1_max = x_test[:, 0].min() - 0.5, x_test[:, 0].max() + 0.5
    x2_min, x2_max = x_test[:, 1].min() - 0.5, x_test[:, 1].max() + 0.5
    
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    Z = lr_model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap='RdBu')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Data and Decision Boundary using scikit-learn')
    plt.show()
    
scikit_learn(X_train, y_train, X_test)
