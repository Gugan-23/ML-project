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
print("Model trained and saved as 'brand_to_elements_model.pkl'.")

# Function to predict element quantities and price for a given brand
def predict_for_brand(brand_name):
    # Load the trained model
    loaded_model = joblib.load('brand_to_elements_model.pkl')
    
    # Create a DataFrame for the input brand
    input_data = pd.DataFrame({'brand': [brand_name]})
    
    # Make predictions
    prediction = loaded_model.predict(input_data)
    
    # Display the prediction results
    element_names = ['Gold (g)', 'Aluminum (g)', 'Silver (g)', 'Carbon (g)', 
                     'Platinum (g)', 'Nickel (g)', 'Lithium (g)', 'Estimated Price (INR)']
    print(f"Predicted values for {brand_name}:")
    for name, value in zip(element_names, prediction[0]):
        print(f"{name}: {value:.2f}")

# Example usage to predict for a brand like 'Apple'
predict_for_brand('HP')
