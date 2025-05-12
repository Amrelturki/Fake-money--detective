import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


# Loading dataset:


# Step 1: Load the dataset (make sure to have the file in the correct path)
data = pd.read_csv('data_banknote_authentication.txt', header=None)

# Check if the data is loaded correctly
print(data.head())  # Print the first few rows
print(data.shape)   # Check the shape of the dataset


# Splitting the Data:


# Step 2: Split dataset into features (X) and labels (y)
X = data.iloc[:, :-1].values  # Features (all columns except the last)
y = data.iloc[:, -1].values   # Labels (last column)

# Check if X and y are correctly extracted
print("Features (X) shape:", X.shape)
print("Labels (y) shape:", y.shape)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Check if X_train and y_train are not empty
print("Training Features (X_train) shape:", X_train.shape)
print("Training Labels (y_train) shape:", y_train.shape)


# Training the model:

# Step 4: Train the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Saving the trained model:

# Step 5: Save the trained model using joblib
joblib.dump(model, 'model.joblib')

print("Model has been trained and saved as 'model.joblib'")


