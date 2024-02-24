Gender Prediction using K-Nearest Neighbors (KNN)

This repository contains Python code for cleaning the data and predicting gender (Male or Female) based on height, weight, and age using the K-Nearest Neighbors (KNN) algorithm. The code includes three different implementations of KNN using different distance metrics: Cartesian, Manhattan, and Minkowski.

## Dependencies

Before running the code, make sure you have the following dependencies installed:
- Python (3.x recommended)
- NumPy
- scikit-learn (for the scikit-learn version)
- pandas (for data preprocessing)

You can install these dependencies using `pip`:
pip install numpy scikit-learn pandas


## Running the Code

### 1. Data Preprocessing

Before running the KNN algorithm, you need to preprocess the data:

- **Step 1**: Prepare your dataset in a CSV file named `ClassData.csv`. This file should contain columns for height, weight, age, and gender.

- **Step 2**: Run the data preprocessing script (Q1 in the code) to perform the following tasks:
  - Load the data from the Excel file and perform data cleaning.
  - Fill missing values with 0.
  - Convert height to inches if necessary.
  - Ensure age is in months.
  - Ensure weight is in pounds.
  - Rename columns for consistency.
  - Save the preprocessed data to `ClassData.csv`.

Run the data preprocessing script with the following command:
python Q1.py

### 2. KNN Models

Next, you can run the KNN models to predict gender based on the preprocessed data:

- Step 1: Run one of the KNN implementations (Q2 or Q3 in the code) based on your preference:

  - For scikit-learn version (Q2), use the following command:
    python Q2.py

  - For custom implementation (Q3), use the following command:
    python Q3.py

- Step 2: The script will prompt you to choose a similarity measurement (C for Cartesian, M for Manhattan, K for Minkowski), order (for Minkowski distance), and K value (number of neighbors).

- **Step 3: The script will display the accuracy of the KNN model on the test data and provide results for different K values and distance measures.

- **Step 4: You can also predict the gender for a specific input record by providing height, weight, and age when prompted.

## Example Usage

Here's an example of how to run the code:
# Step 1: Preprocess the data
python Q1.py

# Step 2: Run the KNN model (scikit-learn version)
python knn_sklearn.py

# Step 3: Choose similarity measurement, order, and K value when prompted
Enter similarity measurement (C for Cartesian, M for Manhattan, K for Minkowski): C
Enter order (1-5): 2
Enter K value (1-5): 3

# Step 4: View accuracy and prediction results
Accuracy: 0.85

# Step 5: Predict gender for a specific input record
Enter height(inches): 65
Enter weight(lbs): 150
Enter age(months): 300
Predicted Gender: Male