import numpy as np
import math

# Load and preprocess the dataset
data = np.genfromtxt('ClassData.csv', delimiter=',', skip_header=1, dtype='str')
X = data[:, :-2]  # Features (excluding 'gender')
y = data[:, -2]   # Target (gender)

# Define a function for label encoding 'gender' (M -> 1, F -> 0)
def label_encode_gender(y):
    return (y == 'M').astype(int)

# Label encode 'gender'
y = label_encode_gender(y)

# Convert the data in X to float type
X = X.astype(float)

# Split the data into training (80%) and testing (20%) sets
np.random.seed(42)  # for reproducibility
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split_idx = int(0.8 * X.shape[0])
train_indices, test_indices = indices[:split_idx], indices[split_idx:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Function to calculate Cartesian distance
def cartesian_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Function to calculate Manhattan distance
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

# Function to calculate Minkowski distance of a given order
def minkowski_distance(point1, point2, order):
    return np.power(np.sum(np.power(np.abs(point1 - point2), order)), 1/order)

# Function to calculate accuracy
def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    return correct_predictions / total_samples

# Function to predict gender using KNN
def knn_predict(X_train, y_train, X_test, k, distance_fn):
    y_pred = np.empty(len(X_test), dtype=int)

    for i in range(len(X_test)):
        distances = [distance_fn(X_test[i], x) for x in X_train]
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:k]
        k_nearest_labels = [y_train[j] for j in k_nearest_indices]
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        y_pred[i] = unique_labels[np.argmax(counts)]

    return y_pred

# Function to calculate confusion matrix
def confusion_matrix(y_true, y_pred):
    unique_labels = np.unique(y_true)
    num_classes = len(unique_labels)
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(y_true)):
        true_label = np.where(unique_labels == y_true[i])[0][0]
        pred_label = np.where(unique_labels == y_pred[i])[0][0]
        matrix[true_label][pred_label] += 1

    return matrix

# Ask the user for similarity measurements, order, and K value
similarity_measure = input("Enter similarity measurement (C for Cartesian, M for Manhattan, K for Minkowski): ")
order = int(input("Enter order (1-5): "))
k_value = int(input("Enter K value (1-5): "))

# Predict gender using KNN
if similarity_measure == 'C':
    distance_fn = cartesian_distance
elif similarity_measure == 'M':
    distance_fn = manhattan_distance
elif similarity_measure == 'K':
    distance_fn = lambda x, y: minkowski_distance(x, y, order)

y_pred = knn_predict(X_train, y_train, X_test, k_value, distance_fn)

# Calculate accuracy
accuracy_score = accuracy(y_test, y_pred)
print(f"Accuracy: {accuracy_score}")

# Display prediction results for different values of K and distance measures for the test data
for k in range(1, 6):
    for measure in ['C', 'M', 'K']:
        if measure == 'C':
            distance_fn = cartesian_distance
        elif measure == 'M':
            distance_fn = manhattan_distance
        elif measure == 'K':
            distance_fn = lambda x, y: minkowski_distance(x, y, order)

        y_pred = knn_predict(X_train, y_train, X_test, k, distance_fn)
        accuracy_score = accuracy(y_test, y_pred)
        print(f"K={k}, Measure={measure}, Accuracy={accuracy_score}")

# Calculate and display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Function to predict gender for a single input record
def predict_gender_single_input(input_data, X_train, y_train, k, distance_fn):
    input_data = np.array(input_data).reshape(1, -1)
    y_pred = knn_predict(X_train, y_train, input_data, k, distance_fn)
    return y_pred[0]

# Predict gender for a single input record
height = float(input("Enter height(inches): "))
weight = float(input("Enter weight(lbs): "))
age = float(input("Enter age(months): "))
input_record = [height, weight, age]
predicted_gender = predict_gender_single_input(input_record, X_train, y_train, k_value, distance_fn)
print(f"Predicted Gender: {'Male' if predicted_gender == 1 else 'Female'}")