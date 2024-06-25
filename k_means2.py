# -*- coding: utf-8 -*-

import os
os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.cluster import KMeans


# Step 1: Load the training and test datasets from the CSV files
def read_csv_data(file_path):
    return pd.read_csv(file_path).values

# Step 2: Z-score normalization
def z_score_normalization(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data

# Step 3: Implement the K-means clustering algorithm
def k_means_clustering(x_train, num_clusters, num_iterations=100):
    num_examples, num_features = x_train.shape

    # Step 3a: Initialize cluster centroids randomly
    np.random.seed(0)
    centroids = x_train[np.random.choice(num_examples, num_clusters, replace=False)]

    for _ in range(num_iterations):
        # Step 3b: Assign each example to the nearest cluster centroid
        distances = np.linalg.norm(x_train[:, np.newaxis] - centroids, axis=2)
        cluster_labels = np.argmin(distances, axis=1)

        # Step 3c: Update cluster centroids
        for cluster_idx in range(num_clusters):
            cluster_examples = x_train[cluster_labels == cluster_idx]
            centroids[cluster_idx] = np.mean(cluster_examples, axis=0)

    return centroids, cluster_labels

# Step 4: Load training dataset and apply Z-score normalization
train_data = read_csv_data("ds2_train.csv")
test_data = read_csv_data("ds2_test.csv")

x_train, y_train = train_data[:, :-1], train_data[:, -1]
x_test, y_test = test_data[:, :-1], test_data[:, -1]

x_train_normalized = z_score_normalization(x_train)

# Step 5: Implement the K-means algorithm and predict cluster labels for the test dataset
num_clusters = 2  # We'll start with 2 clusters, but later we'll tune this hyperparameter
centroids, cluster_labels = k_means_clustering(x_train_normalized, num_clusters)

# Step 6: Load test dataset and apply Z-score normalization
x_test_normalized = z_score_normalization(x_test)

# Step 7: Visualize the clusters using matplotlib
plt.scatter(x_train_normalized[cluster_labels == 0][:, 0], x_train_normalized[cluster_labels == 0][:, 1], label='Cluster 0', marker='o')
plt.scatter(x_train_normalized[cluster_labels == 1][:, 0], x_train_normalized[cluster_labels == 1][:, 1], label='Cluster 1', marker='x')
plt.scatter(centroids[:, 0], centroids[:, 1], label='Cluster Centroids', marker='s', c='red', s=100)
plt.xlabel('Input Variable 1')
plt.ylabel('Input Variable 2')
plt.title('K-means Clustering Visualization')
plt.legend()
plt.show()

# Step 9: Tune the hyperparameter (number of clusters) using the elbow method
def find_optimal_clusters(x_train, max_clusters=10):
    wcss = []
    for num_clusters in range(1, max_clusters + 1):
        centroids, cluster_labels = k_means_clustering(x_train, num_clusters)
        wcss.append(np.sum(np.min(np.linalg.norm(x_train[:, np.newaxis] - centroids, axis=2), axis=1)))

    return wcss

wcss = find_optimal_clusters(x_train_normalized, max_clusters=10)

# Plot the WCSS (Within-cluster sum of squares) against the number of clusters
plt.plot(range(1, 11), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# The "elbow" point represents the optimal number of clusters
optimal_clusters = 2  # You can select the elbow point manually or use an automatic method

# Re-train the K-means algorithm with the optimal number of clusters
optimal_centroids, optimal_cluster_labels = k_means_clustering(x_train_normalized, optimal_clusters)

# Step 10: Evaluate the K-means algorithm on the test dataset
predicted_test_labels = np.argmin(np.linalg.norm(x_test_normalized[:, np.newaxis] - optimal_centroids, axis=2), axis=1)

# Step 11: Visualize the clusters on the test dataset
plt.scatter(x_test_normalized[predicted_test_labels == 0][:, 0], x_test_normalized[predicted_test_labels == 0][:, 1], label='Cluster 0', marker='o')
plt.scatter(x_test_normalized[predicted_test_labels == 1][:, 0], x_test_normalized[predicted_test_labels == 1][:, 1], label='Cluster 1', marker='x')
plt.scatter(optimal_centroids[:, 0], optimal_centroids[:, 1], label='Cluster Centroids', marker='s', c='red', s=100)
plt.xlabel('Input Variable 1')
plt.ylabel('Input Variable 2')
plt.title('K-means Clustering on Test Dataset')
plt.legend()
plt.show()

# Print the best hyperparameter value and the corresponding evaluation metric score
print(f"Optimal number of clusters: {optimal_clusters}")

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
  #  precision = precision_score(y_test, y_pred_test,average='weighted',pos_label='positive')
    recall = recall_score(y_test, y_pred_test,average='weighted')
  #  f1 = f1_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    
    print("Accuracy using scikit-learn:", accuracy)
 #   print("Precision using scikit-learn:", precision)
    print("Recall using scikit-learn:", recall)
  #  print("F1 Score using scikit-learn:", f1)
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



