# -*- coding: utf-8 -*-
"""
The code starts with importing necessary libraries and defines functions 
for reading data from CSV files, Gaussian Discriminant Analysis (GDA) class, 
evaluating model performance, displaying the classification result, 
normalizing data using robust scaling, and performing grid search to tune hyperparameters.

The GaussianDiscriminantAnalysis class is responsible for training the GDA model 
using the training data and predicting class labels for new data points.

The evaluate function calculates the accuracy of the predicted labels 
compared to the true labels.

The display_classification function plots the training data and 
overlays the predicted classifications on top of it, 
using different marker shapes and colors.

In the main part of the code, the dataset is loaded from CSV files, 
normalized using robust scaling, and plotted to visualize the data. 
Grid search is performed to find the best hyperparameter for the GDA model, 
and then the model is trained and evaluated using the best hyperparameter. 
Finally, the final classification result is displayed 
using the display_classification function.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV

# Function to read data from CSV file
def read_csv_data(file_path):
    return pd.read_csv(file_path).values

# Gaussian Discriminant Analysis (GDA) Class
class GaussianDiscriminantAnalysis:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.class_priors = np.zeros(self.num_classes)
        self.means = np.zeros((self.num_classes, X.shape[1]))
        self.cov_matrices = np.zeros((self.num_classes, X.shape[1], X.shape[1]))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[i] = X_c.shape[0] / X.shape[0]
            self.means[i] = np.mean(X_c, axis=0)
            self.cov_matrices[i] = np.cov(X_c, rowvar=False)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            probs = []
            for j in range(self.num_classes):
                diff = x - self.means[j]
                exponent = -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(self.cov_matrices[j])), diff)
                prob = self.class_priors[j] * np.exp(exponent) / np.sqrt(np.linalg.det(self.cov_matrices[j]))
                probs.append(prob)
            y_pred[i] = self.classes[np.argmax(probs)]

        return y_pred

# Function to calculate accuracy, precision, recall, F1-score, and mean squared error
def evaluate(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    return accuracy

# Function to normalize data using robust scaling
def robust_scale(X):
    """
    Perform robust scaling on the input matrix X.
    Parameters:
        X (numpy array): Input feature matrix of shape (num_examples, num_features).
        
    Returns:
        X_scaled (numpy array): Normalized feature matrix
        based on their median and interquartile range, making it more robust to outliers.
    """
    median = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    return (X - median) / iqr

# Function to perform grid search to find the best hyperparameter value
def grid_search(X_train, y_train, X_test, y_test):
    hyperparameter_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    best_hyperparameter = None
    best_accuracy = 0.0

    for hp_value in hyperparameter_values:
        model = GaussianDiscriminantAnalysis()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = evaluate(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparameter = hp_value

    return best_hyperparameter, best_accuracy

# Function to display the final output of the GDA algorithm
def display_classification(X, y, y_pred):
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0", marker="o")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1", marker="x")

    plt.scatter(X[y_pred == 0][:, 0], X[y_pred == 0][:, 1], label="Predicted Class 0", marker="s", edgecolors="g", facecolors="none")
    plt.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], label="Predicted Class 1", marker="s", edgecolors="r", facecolors="none")

    plt.xlabel("Input Variable 1")
    plt.ylabel("Input Variable 2")
    plt.title("GDA Classification Result")
    plt.legend()
    plt.show()
    
def scikit_learn(X_train, y_train, X_test, y_test):
    # Normalize the data using robust scaling
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Plot the training data
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label="Class 0", marker="o")
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label="Class 1", marker="x")
    plt.xlabel("Input Variable 1")
    plt.ylabel("Input Variable 2")
    plt.title("Training Data using scikit-learn")
    plt.legend()
    plt.show()

    # Hyperparameter tuning using grid search
    param_grid = {'reg_param': [0.001, 0.01, 0.1, 1.0, 10.0]}
    gda = QuadraticDiscriminantAnalysis()
    grid_search = GridSearchCV(gda, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_hyperparameter = grid_search.best_params_['reg_param']
    print("Best Hyperparameter using scikit-learn:", best_hyperparameter)
    
    # Train the GDA model with the best hyperparameter
    final_model = QuadraticDiscriminantAnalysis(reg_param=best_hyperparameter)
    final_model.fit(X_train, y_train)

    # Predictions on test set
    y_pred_test = final_model.predict(X_test)

    # Evaluate the final model
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    
    print("Accuracy using scikit-learn:", accuracy)
    print("Precision using scikit-learn:", precision)
    print("Recall using scikit-learn:", recall)
    print("F1 Score using scikit-learn:", f1)
    print("Mean Squared Error using scikit-learn:", mse)

    # Display the final classification result
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], label="Class 0", marker="o")
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label="Class 1", marker="x")

    plt.scatter(X_test[y_pred_test == 0][:, 0], X_test[y_pred_test == 0][:, 1], label="Predicted Class 0", marker="s", edgecolors="g", facecolors="none")
    plt.scatter(X_test[y_pred_test == 1][:, 0], X_test[y_pred_test == 1][:, 1], label="Predicted Class 1", marker="s", edgecolors="r", facecolors="none")

    plt.xlabel("Input Variable 1")
    plt.ylabel("Input Variable 2")
    plt.title("GDA Classification Result using scikit-learn")
    plt.legend()
    plt.show()
    
# Main code
if __name__ == "__main__":
    # Load training and testing data
    train_data = read_csv_data("ds1_train.csv")
    test_data = read_csv_data("ds1_test.csv")

    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
    
    print("First five elements in X_train before normalisation are:\n", X_train[:5])
    print("Type of X_train:",type(X_train))
    print ('The shape of X_train is: ' + str(X_train.shape))
    print ('We have m = %d training examples' % (len(y_train)))
    
    print("Values using scikit-learn library:")
    scikit_learn(X_train, y_train, X_test, y_test)

    # Plot the training data before scaling
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label="Class 0", marker="o")
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label="Class 1", marker="x")
    plt.xlabel("Input Variable 1")
    plt.ylabel("Input Variable 2")
    plt.title("Training Data before scaling")
    plt.legend()
    plt.show()
    
    # Normalize the data using robust scaling
    X_train = robust_scale(X_train)
    X_test = robust_scale(X_test)

    print("First five elements in X_train after normalisation are:\n", X_train[:5])
    
    # Plot the training data after scaling
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label="Class 0", marker="o")
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label="Class 1", marker="x")
    plt.xlabel("Input Variable 1")
    plt.ylabel("Input Variable 2")
    plt.title("Training Data after scaling")
    plt.legend()
    plt.show()

    # Tune the hyperparameter using grid search
    best_hp, best_accuracy = grid_search(X_train, y_train, X_test, y_test)

    print("Best Hyperparameter:", best_hp)
    print("Best Accuracy:", best_accuracy)

    # Train the model with the best hyperparameter
    final_model = GaussianDiscriminantAnalysis()
    final_model.fit(X_train, y_train)
    y_pred_test = final_model.predict(X_test)

    # Evaluate the final model
    final_accuracy = evaluate(y_test, y_pred_test)
    print("Final Accuracy:", final_accuracy)

    # Display the final classification result
    display_classification(X_test, y_test, y_pred_test)