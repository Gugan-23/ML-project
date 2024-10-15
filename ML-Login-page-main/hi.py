import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

# Load the dataset
data = pd.read_excel(r'D:\ML project\updated_material_usage_dataset_with_prices_inr.xlsx')

# Separate features and target
X = data[['brand']]  # Use only the brand as the input feature
y = data[['Gold (g)', 'Aluminum (g)', 'Silver (g)', 'Carbon (g)', 
          'Platinum (g)', 'Nickel (g)', 'Lithium (g)']]  # Elements as target variables

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessor for the brand with handle_unknown set to 'ignore'
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['brand'])  # Handle unknown categories
    ])

# Create a pipeline with preprocessing and the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(LinearRegression()))
])

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'brand_to_elements_model.pkl')

print("Model trained and saved successfully.")

# Function to predict material usage based on brand input
def predict_material_usage(brand_name):
    # Check if the brand is known; if not, predict using average values
    if brand_name not in X_train['brand'].values:
        avg_values = y_train.mean().values  # Use the mean values of the training set
        print(f"Brand '{brand_name}' not found. Predicting using average values: {avg_values}")
        return avg_values
    else:
        # Create a DataFrame for the new input
        input_data = pd.DataFrame({'brand': [brand_name]})
        # Predict using the trained model
        prediction = model.predict(input_data)
        return prediction[0]

# Example usage
while(True):
    brand_input = input("Enter the brand name: ")
    predicted_usage = predict_material_usage(brand_input)
    print(f"Predicted material usage for brand '{brand_input}': {predicted_usage}")
