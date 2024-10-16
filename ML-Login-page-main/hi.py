import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
data = pd.read_excel(r'D:\ML project\updated_material_usage_dataset_with_prices_inr.xlsx')

# Define features and target variables
X = data[['brand']]  
y = data[['Gold (g)', 'Aluminum (g)', 'Silver (g)', 'Carbon (g)', 
          'Platinum (g)', 'Nickel (g)', 'Lithium (g)', 'Estimated Price (INR)']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessor for the 'brand' column with handle_unknown='ignore'
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['brand'])  # One-hot encode the 'brand' column
    ]
)

# Create a pipeline with preprocessing and the MultiOutputRegressor model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(LinearRegression()))
])

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'brand_to_elements_model.pkl')

# Make predictions on the test set
predictions = model.predict(X_test)

# Print the predicted values
for i, prediction in enumerate(predictions):
    print(f"Prediction {i + 1}: {prediction}")

print("Model trained and saved successfully.")
