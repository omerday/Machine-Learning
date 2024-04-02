###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    min_x = np.min(X)
    max_x = np.max(X)
    avg_x = np.average(X)

    X = (X - avg_x) / (max_x - min_x)

    min_y = np.min(y)
    max_y = np.max(y)
    avg_y = np.average(y)

    y = (y - avg_y) / (max_y - min_y)
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    if len(X.shape) == 1:
        X = np.reshape(X, (-1, 1))
    ones = np.ones((X.shape[0], 1))
    X = np.concatenate([ones, X], axis=1)

    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    m = y.shape[0]
    h_theta = X * theta
    h_theta_sum = np.sum(h_theta, axis=1)
    error_squared = (y - h_theta_sum)**2
    J = (1/(2*m)) * np.sum(error_squared)
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = y.shape[0]
    for i in range(num_iters):
        theta_temp = theta.copy()
        for j in range(len(theta)):
            h_theta = X * theta
            h_theta_sum = np.sum(h_theta, axis=1)
            error = (h_theta_sum - y) * X[:,j]
            theta_temp[j] = theta[j] - (alpha/m) * np.sum(error)
        theta = theta_temp
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    X_t = np.matrix.transpose(X)
    X_mult = np.matmul(X_t, X)
    X_mult_inverse = np.linalg.inv(X_mult)
    X_mult_inverse_mult = np.matmul(X_mult_inverse, X_t)
    pinv_theta = np.matmul(X_mult_inverse_mult, y)
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = y.shape[0]
    i = 0
    while i < num_iters and (len(J_history) < 2 or J_history[-2] - J_history[-1] > 1E-8):
        theta_temp = theta.copy()
        for j in range(len(theta)):
            h_theta = X * theta
            h_theta_sum = np.sum(h_theta, axis=1)
            error = (h_theta_sum - y) * X[:,j]
            theta_temp[j] = theta[j] - (alpha/m) * np.sum(error)
        theta = theta_temp
        J_history.append(compute_cost(X, y, theta))
        i += 1
    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    np.random.seed(42)
    theta = np.random.random(size=X_train.shape[1])
    for alpha in alphas:
        theta_alpha, J_history = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta_alpha)
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    n = X_train.shape[1]

    while len(selected_features) < 5:
        cheapest = float('inf')
        cheapest_index = None
        for i in range(n):
            if i not in selected_features:
                selected_features.append(i)

                X_train_temp = X_train[:, selected_features]
                X_train_temp = apply_bias_trick(X_train_temp)

                X_val_temp = X_val[:, selected_features]
                X_val_temp = apply_bias_trick(X_val_temp)

                np.random.seed(42)
                theta = np.random.random(size=X_train_temp.shape[1])

                theta, _ = efficient_gradient_descent(X_train_temp, y_train, theta, best_alpha, iterations)
                price = compute_cost(X_val_temp, y_val, theta)

                if price < cheapest:
                    cheapest = price
                    cheapest_index = i

                selected_features.remove(i)
        selected_features.append(cheapest_index)

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    columns = df.columns
    for feature in columns:
        df_poly[feature+"^2"] = pow(df_poly[feature], 2)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            df_poly[f"{columns[i]}*{columns[j]}"] = df_poly[columns[i]] * df_poly[columns[j]]
    return df_poly
#%%
