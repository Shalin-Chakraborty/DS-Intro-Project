# -*- coding: utf-8 -*-

# importing libraries    
import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np      
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.cluster import KMeans

# load dataset
def read_csv_file(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y

train_file_path = 'ds1_train.csv'
test_file_path = 'ds1_test.csv'
x_train, y_train = read_csv_file(train_file_path)
x_test, y_test = read_csv_file(test_file_path)

print("First five elements in x1_train before normalisation are:\n", x_train[:5])
print("Type of x1_train:",type(x_train))
print ('The shape of x1_train is: ' + str(x_train.shape))
print ('We have m = %d training examples' % (len(y_train)))

plt.plot(x_train,y_train)
plt.show()

# Function to perform Robust Scaling on the input matrix X
def robust_scaling(X):
    median_vals = np.median(X, axis=0)
    quartile_1 = np.percentile(X, 25, axis=0)
    quartile_3 = np.percentile(X, 75, axis=0)
    IQR = quartile_3 - quartile_1
    X_scaled = (X - median_vals) / IQR
    return X_scaled

x_train=robust_scaling(x_train)
x_test=robust_scaling(x_test)

plt.plot(x_train,y_train,'o')
plt.show()

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Function to calculate Within-Cluster Sum of Squares (WCSS)
def calculate_wcss(X, centroids, cluster_assignments):
    wcss = 0
    for i in range(len(X)):
        cluster_idx = cluster_assignments[i]
        wcss += euclidean_distance(X[i], centroids[cluster_idx]) ** 2
    return wcss

# Function to implement K-Means clustering
def kmeans_clustering(X, num_clusters, max_iterations=100):
    num_examples, num_features = X.shape
    
    # Randomly initialize centroids
    centroids_idx = np.random.choice(num_examples, num_clusters, replace=False)
    centroids = X[centroids_idx]
    
    prev_cluster_assignments = None
    
    for _ in range(max_iterations):
        # Assign each example to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)
        
        # Check for convergence
        if np.array_equal(cluster_assignments, prev_cluster_assignments):
            break
        
        prev_cluster_assignments = cluster_assignments
        
        # Update centroids
        new_centroids = np.array([X[cluster_assignments == i].mean(axis=0) for i in range(num_clusters)])
        centroids = new_centroids
        
    return cluster_assignments, centroids

# Function to find the optimal number of clusters using the Elbow Method
def find_optimal_clusters(X, max_clusters=10):
    wcss_values = []
    for num_clusters in range(1, max_clusters+1):
        cluster_assignments, centroids = kmeans_clustering(X, num_clusters)
        wcss = calculate_wcss(X, centroids, cluster_assignments)
        wcss_values.append(wcss)
        
    plt.plot(range(1, max_clusters+1), wcss_values)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method: Optimal Number of Clusters')
    plt.show()
    
# Find the optimal number of clusters using the Elbow Method
find_optimal_clusters(x_train)

# Choose the optimal number of clusters (from the elbow plot)
num_clusters = 2

def tune_hyperparameter(X, max_clusters=10):
    wcss_values = []
    for num_clusters in range(1, max_clusters+1):
        cluster_assignments, centroids = kmeans_clustering(X, num_clusters)
        wcss = calculate_wcss(X, centroids, cluster_assignments)
        wcss_values.append(wcss)
        
    # Plot the WCSS values for different values of K
    plt.plot(range(1, max_clusters+1), wcss_values)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method: Hyperparameter Tuning')
    plt.show()

    # Find the optimal value of K using the Elbow Method
    diff_wcss = np.diff(wcss_values)
    elbow_point = np.argmax(diff_wcss) + 1  # Adding 1 as the index is 0-based
    optimal_k = elbow_point + 1  # Adding 1 to get the optimal value of K
    return optimal_k

optimal_k = tune_hyperparameter(x_train)
optimal_k = 2

print(f"Optimal value of K: {optimal_k}")

# Function to plot the data with different markers for positive and negative examples
def plot_data_with_markers(X, y, centroids=None):
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], marker='o', label='Negative')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='x', label='Positive')
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='s', c='red', label='Centroids')
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.legend()
    plt.title('Data with Markers')
    plt.show()
"""
# Train K-Means clustering on the training data
cluster_assignments, centroids = kmeans_clustering(x_train, num_clusters)
"""
#Train K-Means clustering on the training data with the optimal value of K
cluster_assignments, centroids = kmeans_clustering(x_train, optimal_k)


# Predict the values in y_test
test_distances = np.linalg.norm(x_test[:, np.newaxis] - centroids, axis=2)
predicted_labels = np.argmin(test_distances, axis=1)

# Predict the values in y_test using the optimal K-Means model
test_distances = np.linalg.norm(x_test[:, np.newaxis] - centroids, axis=2)
predicted_labels = np.argmin(test_distances, axis=1)

# Visualize the clusters and data points
plot_data_with_markers(x_train, y_train, centroids)

# Evaluate the performance using suitable evaluation metrics (e.g., accuracy, F1-score, etc.)
# For demonstration purposes, let's calculate accuracy here.
accuracy = np.mean(predicted_labels == y_test)
print(f"Accuracy on Test Set with K={optimal_k}: {accuracy*100:.2f}%")

def scikit_learn(x_train,y_train,x_test,y_test):
    #finding optimal number of clusters using the elbow method  
    wcss_list= []  #Initializing the list for the values of WCSS  
      
    #Using for loop for iterations from 1 to 10.  
    for i in range(1, 10):  
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42, n_init='auto')  
        kmeans.fit(x_train)  
        wcss_list.append(kmeans.inertia_)  
    plt.plot(range(1, 10), wcss_list)  
    plt.title('The Elbow Method Graph using scikit-learn')  
    plt.xlabel('Number of clusters(k)')  
    plt.ylabel('wcss_list')  
    plt.show()  
    
    # Predictions on test set
    y_pred_test = kmeans.predict(x_test)

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
    
    #training the K-means model on a dataset  
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state= 42, n_init='auto')  
    y_test= kmeans.fit_predict(x_test)  
    
    plt.scatter(x_train[y_train == 0][:, 0], x_train[y_train == 0][:, 1], marker='o', label='Negative')
    plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], marker='x', label='Positive')
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='s', c='red', label='Centroids')
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.legend()
    plt.title('Data with Markers using scikit-learn')
    plt.show()
    
scikit_learn(x_train, y_train, x_test, y_test)
