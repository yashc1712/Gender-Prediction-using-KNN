import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv("ClassData.csv")

# Encode 'Gender(M/F)' as numerical values (M -> 1, F -> 0)
data['Gender(M/F)'] = data['Gender(M/F)'].map({'M': 1, 'F': 0})

# Specify feature names
feature_names = ['Height', 'Weight', 'Age(Months)']

# Split the data into features (X) and labels (y)
X = data[feature_names]
y = data['Gender(M/F)']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ask the user for similarity measurements, order, and K value
similarity_measure = input("Enter similarity measurement (C for Cartesian, M for Manhattan, K for Minkowski): ")
order = int(input("Enter order (1-5): "))
k_value = int(input("Enter K value (1-5): "))

# Create the KNN classifier with the specified parameters
if similarity_measure == 'C':
    knn = KNeighborsClassifier(n_neighbors=k_value, p=2)  # p=2 for Cartesian (Euclidean) distance
elif similarity_measure == 'M':
    knn = KNeighborsClassifier(n_neighbors=k_value, p=1)  # p=1 for Manhattan distance
elif similarity_measure == 'K':
    knn = KNeighborsClassifier(n_neighbors=k_value, p=order)  # Minkowski distance with custom order

# Fit the classifier on the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy}")

# Display prediction results for different values of K and distance measures for the test data
for k in range(1, 6):
    for measure in ['C', 'M', 'K']:
        if measure == 'C':
            knn = KNeighborsClassifier(n_neighbors=k, p=2)
        elif measure == 'M':
            knn = KNeighborsClassifier(n_neighbors=k, p=1)
        elif measure == 'K':
            knn = KNeighborsClassifier(n_neighbors=k, p=order)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"K={k}, Measure={measure}, Accuracy={accuracy}")

# Calculate and display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Predict gender for a specific input record
height = float(input("Enter height(inches): "))
weight = float(input("Enter weight(lbs): "))
age = float(input("Enter age(months): "))
input_record = [[height, weight, age]]
predicted_gender = knn.predict(input_record)

# Map numerical prediction back to 'M' or 'F'
predicted_gender_label = 'Male' if predicted_gender[0] == 1 else 'Female'

# Display the predicted gender
print(f"Predicted Gender: {predicted_gender_label}")