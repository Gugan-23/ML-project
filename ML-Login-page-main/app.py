import pandas as pd
from flask import Flask, request, jsonify,send_file
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import os
from flask import Flask, request, jsonify
import pandas as pd
import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
model2=joblib.load('brand_to_elements_model.pkl')
app = Flask(__name__)

from flask_cors import CORS

app = Flask(__name__)


import os
def index():
    return '<h1>Welcome to the Plotting App!</h1><a href="/plot">View Plot</a>'

@app.route('/plot')
def plot():
    # Load the dataset
    data = pd.read_excel(r"C:\Users\Administrator\Downloads\archive (2)\data1.xlsx")
    df = pd.DataFrame(data)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Product', palette='viridis')
    plt.title('Count of Each Product Class')
    plt.xlabel('Product Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Save the plot
    plot_path = 'correlation_graph.png'
    plt.savefig(plot_path)
    plt.close()

    return send_file(plot_path)  # Send the plot image to the client

data_path = r"C:\Users\Administrator\Downloads\archive (2)\data1.xlsx"
df = pd.read_excel(data_path)

# Create a route for the correlation matrix
@app.route('/correlation', methods=['GET'])
@app.route('/correlation', methods=['GET'])
def correlation():
    print('hi')
    try:
        print("h1")
        # Load the dataset
        data_path = r"C:\Users\Administrator\Downloads\archive (2)\data1.xlsx"
        df = pd.read_excel(data_path)
        print(df.head())
        
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_df.corr()
        
        print('hi,', correlation_matrix)
        # Convert to a dictionary for JSON response
        return jsonify(correlation_matrix.to_dict()), 200

    except Exception as e:
        print("Error:", str(e))  # Log the error for debugging
        return jsonify({'error': str(e)}), 500

# Fetch User Profile

import threading
import tkinter as tk
def open_tkinter_window():
    import multiple 

@app.route('/open-window', methods=['GET'])
def open_window():
    import multiple 
    # Open tkinter window in a separate thread
    threading.Thread(target=open_tkinter_window).start()
    return "Window opened"

def calculate_correlation_matrix(data):
    # Calculate the correlation matrix, automatically handles NaN values using method='pearson'
    correlation = data.corr(method='pearson')
    # Convert the correlation DataFrame to a dictionary
    return correlation.to_dict()

@app.route('/correlation')
def get_correlation():
    # Load your data
    data = pd.read_excel(r'C:\Users\Administrator\Downloads\archive (2)\data1.xlsx')

    # Calculate the correlation matrix
    correlation_result = calculate_correlation_matrix(data)

    # Replace NaN values with None for JSON compatibility
    correlation_result = correlation_result.where(pd.notnull(correlation_result), None).astype(object).to_dict()

    return jsonify(correlation_result)
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

from forex_python.converter import CurrencyRates
import numpy as np
def get_exchange_rates(base_currency):
    api_url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json().get('rates')
        else:
            print(f"Error fetching currency rates: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching currency rates: {e}")
        return None

def convert_currency(amount, rates, to_currency):
    if to_currency in rates:
        return amount * rates[to_currency]
    else:
        print(f"Currency {to_currency} not found in exchange rates.")
        return None
    
data = pd.read_excel(r'C:\Users\Administrator\Downloads\archive (2)\data1.xlsx')
X = data.drop(['newprice', 'price', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1, errors='ignore')
y = data['price']
unique_brands = data['brand'].unique()
X['Ram'] = X['Ram'].str.replace('GB', '').replace('', np.nan)  
X['Ram'] = X['Ram'].fillna(0)  
X['Ram'] = X['Ram'].astype(int)  
def convert_ROM_to_GB(rom):
    if isinstance(rom, str) and 'TB' in rom:
        return int(rom.replace('TB', '').strip()) * 1024  
    elif isinstance(rom, str) and 'GB' in rom:
        return int(rom.replace('GB', '').strip()) 
    else:
        return 0  

X['ROM'] = X['ROM'].apply(convert_ROM_to_GB)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))  
    label_encoders[column] = le


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()  
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
def encode_with_unknown(le, value):
    """ Encode the value using LabelEncoder, and return a special code for unseen labels. """
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return len(le.classes_)  

def full_process(new_input):
    data = pd.read_excel(r'C:\Users\Administrator\Downloads\archive (2)\data1.xlsx')
    df = pd.DataFrame(data)
    df = df[['brand', 'name', 'price', 'Product']]

    
    le_company = LabelEncoder()
    df['brand'] = le_company.fit_transform(df['brand'].astype(str))
    
    le_name = LabelEncoder()
    df['name'] = le_name.fit_transform(df['name'].astype(str))
    
    
    X_price = df[['brand', 'name']]
    y_price = df['price']
    X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

    
    model_price = RandomForestRegressor()
    model_price.fit(X_train_price, y_train_price)

    # Predict and evaluate price
    y_pred_price = model_price.predict(X_test_price)
    mse = mean_squared_error(y_test_price, y_pred_price)
    mae = mean_absolute_error(y_test_price, y_pred_price)
    r2 = r2_score(y_test_price, y_pred_price)

    print(f"Price Prediction - Mean Squared Error: {mse:.2f}")
    print(f"Price Prediction - Mean Absolute Error: {mae:.2f}")
    print(f"Price Prediction - R-squared: {r2:.2f}")
    

    # Clean the DataFrame for Product prediction
    df_clean = df.dropna(subset=['Product'])
    X_product = df_clean[['brand', 'name']]
    y_clean_product = df_clean['Product']
    
    # Use RandomForestClassifier for categorical output
    model_product = RandomForestClassifier()
    X_train_product, X_test_product, y_train_product, y_test_product = train_test_split(X_product, y_clean_product, test_size=0.2, random_state=42)
    model_product.fit(X_train_product, y_train_product)

    # Predict and evaluate Product
    y_pred_product = model_product.predict(X_test_product)
    accuracy = accuracy_score(y_test_product, y_pred_product)
    print(f"Product Prediction - Accuracy: {accuracy * 100:.2f}%")

    # Encode the new input for prediction
    new_company_encoded = le_company.transform([new_input['brand']])[0]
    new_name_encoded = encode_with_unknown(le_name, new_input['name'])

    # Check if the new input exists in the original DataFrame
    match = df[(df['brand'] == new_company_encoded) & (df['name'] == new_name_encoded)]
    
    if not match.empty:
        existing_price = match['price'].values[0]
        existing_product = match['Product'].values[0]
        print(f"Existing price: {existing_price:.2f}")
        print(f"Existing Product: {existing_product}")
    else:
        keywords = ['Printer', 'Scanner', 'Copier']
        if any(keyword.lower() in new_input['name'].lower() for keyword in keywords):
            print(f"The input '{new_input['name']}' matches the category: Office Appliances (Printer, Scanner, Copier).")
            
            
        else:
            # Predict price and product if not matched
            new_data = pd.DataFrame({
                'brand': [new_company_encoded],
                'name': [new_name_encoded]
            })
        
            predicted_price = model_price.predict(new_data)
            predicted_product = model_product.predict(new_data)

            print(f"Predicted price: {predicted_price[0]:.2f}")
            print(f"Predicted Product: {predicted_product[0]}")
  
def predict_product_type(er_specifications,display_size,data):
    print(er_specifications)
    print('ram',er_specifications['Ram'])
        
    if display_size==None:
        full_process(er_specifications)
        return "Unknown"
        
    if display_size<4:

        return "watch"
    if display_size < 7:
        return "Phone"
    elif 7 <= display_size < 15:
        return "Phablet"  # Optional: for larger phones
        
    elif 15 <= display_size < 20:
        return "Laptop"
    elif 20 <= display_size < 50:
        return "TV"
    else:
        return "Unknown"
        
        return 0
def convert_GPU_to_numeric(gpu):
    if pd.isna(gpu):  # Check if the value is NaN
        return 0  # or any other appropriate value for NaN
    if isinstance(gpu, int):  # If the value is already an int, return it as is
        return gpu
    if isinstance(gpu, str):  # Ensure gpu is a string before checking
        if 'GTX' in gpu:
            return 1  
        elif 'RTX' in gpu:
            return 2  
        elif 'Intel' in gpu or 'Iris' in gpu:
            return 3  
        elif 'AMD' in gpu:
            return 4  
        elif 'Apple' in gpu:
            return 5  
        elif 'NVIDIA' in gpu:
            return 6  
    return 0  # Return 0 for any other case

X['GPU'] = X['GPU'].apply(convert_GPU_to_numeric)

categorical_columns = ['brand', 'name', 'processor', 'CPU', 'yearuser', 'Ram_type', 'ROM_type', 'OS']
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2=None
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Return the global R² score

def fill_missing_features(user_input, X_train):
    important_features = ['OS', 'Ram', 'ROM', 'brand', 'CPU']  
    if user_input['Ram'] is None:
        print("Ram value is None. Unable to predict price.")
        return None  

    for column in X_train.columns:
        if column not in user_input or pd.isna(user_input[column]):
            if column in important_features:
                continue  
            if X_train[column].dtype == 'object':
                user_input[column] = X_train[column].mode()[0]
            else:
                user_input[column] = X_train[column].mean()
    return user_input
import pandas as pd

def find_price(user_input, data):
    
    conditions = pd.Series(True, index=data.index)

    for key, value in user_input.items():
        
        if value is not None:
            conditions &= (data[key] == value)  
    filtered_df = data[conditions]
    
    if not filtered_df.empty:
        return filtered_df['newprice'].values[0]  
    else:
        return None  

er_specifications = {
    'brand': 'Samsung',
    'name': 'Airdopes 381 Sunburn Edition with up to 20 Hours Playtime',
    'processor': None,
    'CPU': None,
    'Ram': '32GB',
    'Ram_type': 'LPDDR5X',
    'ROM': '1TB',
    'ROM_type': 'SSD',
    'OS': 'Windows 11 OS',
    'GPU': 'Intel Integrated Iris Xe',
    'display_size': 17,
    'resolution_width': 1920,
    'resolution_height': 1080,
    'warranty': None,  # This will be ignored
    'yearuser': None,  # This will be ignored
    'avg_rating': None,  # This will be ignored
    '5G_or_not': None,  # This will be ignored
    'processor_brand': None,  # This will be ignored
    'num_cores': None,  # This will be ignored
    'processor_speed': None,  # This will be ignored
    'battery_capacity': None,  # This will be ignored
    'fast_charging_available': None,  # This will be ignored
    'fast_charging': None,  # This will be ignored
    'refresh_rate': None,  # This will be ignored
    'num_rear_cameras': None,  # This will be ignored
    'primary_camera_rear': None,  # This will be ignored
    'primary_camera_front': None,  # This will be ignored
    'extended_memory_available': None  # This will be ignored
}
def update_specifications(specifications, brand, name, display_size):
    print(specifications, brand, name, display_size)
    if brand:
        specifications['brand'] = brand
    if name:
        specifications['name'] = name
    if display_size:
        try:
            specifications['display_size'] = int(display_size)
            print('displaysize')
            find_price(er_specifications,data)
        except ValueError:
            print("Invalid input for display size. Keeping the current value.")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from flask import Flask, request, jsonify

app = Flask(__name__)

def run_random_forest_model():
    # Assuming the data comes from the request
    
    # Load data from Excel file
    data = pd.read_excel(r'C:\Users\Administrator\Downloads\archive (2)\data1.xlsx')

    # Prepare features and target variable
    X = data.drop(['newprice', 'price', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1, errors='ignore')
    y = data['price']

    # Convert RAM column to integer values
    X['Ram'] = X['Ram'].str.replace('GB', '').replace('', np.nan)  
    X['Ram'] = X['Ram'].fillna(0).astype(int)  

    # Convert ROM column to GB
    def convert_ROM_to_GB(rom):
        if isinstance(rom, str) and 'TB' in rom:
            return int(rom.replace('TB', '').strip()) * 1024  
        elif isinstance(rom, str) and 'GB' in rom:
            return int(rom.replace('GB', '').strip()) 
        else:
            return 0  

    X['ROM'] = X['ROM'].apply(convert_ROM_to_GB)

    # Encode categorical variables
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))  
        label_encoders[column] = le

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate R² score
    r2 = r2_score(y_test, y_pred)

    return jsonify({'r2_score': r2})

CORS(app) 

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()  # Get the JSON data from the request
    brand = data.get('brand', '')
    name = data.get('name', '')
    display_size = data.get('display_size', '')

    print('hello',er_specifications, brand, name, display_size)
    # Update specifications based on input
    run_random_forest_model()
    
    update_specifications(er_specifications, brand, name, display_size)
    product_type = predict_product_type(er_specifications,er_specifications['display_size'],data)
    print(f"The predicted product type is: {product_type}")

    
    # Return the updated specifications as a JSON response
    return jsonify(product_type)

def get_brand_details(brand_name):
    
    samsung_data = pd.read_excel(r"D:\ML project\updated_material_usage_dataset_with_prices_inr.xlsx")
    samsung_data.columns = samsung_data.columns.str.strip()  

    print("Dataset Columns:", samsung_data.columns)

    # Check if the dataset is not empty
    if not samsung_data.empty:
        # Verify if the 'brand' column exists
        if 'brand' in samsung_data.columns:
            # Filter for the given brand name
            filtered_data = samsung_data[samsung_data['brand'] == brand_name]
            if not filtered_data.empty:
                print(f"Details for {brand_name}:\n", filtered_data)
                return filtered_data
            else:
                print(f"No details found for {brand_name} products.")
                return None
        else:
            print("Column 'brand' not found in the dataset.")
            return None
    else:
        print("Dataset is empty.")
        return None
if (er_specifications['brand']):
    infinix_details = get_brand_details(er_specifications['brand'])
def predict_price(user_input):
    if 'Ram' in user_input:
        user_input['Ram'] = int(user_input['Ram'].replace('GB', ''))
    if 'ROM' in user_input:
        user_input['ROM'] = convert_ROM_to_GB(user_input['ROM'])
    if 'GPU' in user_input:
        user_input['GPU'] = convert_GPU_to_numeric(user_input['GPU'])
    
    matched_price = find_price(user_input,data)
    
    if matched_price is not None:
        return matched_price
    
    user_input = fill_missing_features(user_input, X_train)

    for column in categorical_columns:
        if column in user_input:
            if user_input[column] in label_encoders[column].classes_:
                user_input[column] = label_encoders[column].transform([user_input[column]])[0]
            else:
                user_input[column] = -1

    user_input_df = pd.DataFrame([user_input])[X_train.columns]

    predicted_price = model.predict(user_input_df)
    
    return predicted_price[0]


product_type = predict_product_type(er_specifications,er_specifications['display_size'],data)
print(f"The predicted product type is: {product_type}")

price = find_price(er_specifications,data)

if price is not None:
    print(f"The price of the specified laptop is: ₹{price:.2f}")
    rates = get_exchange_rates("INR")
    if rates:
        print("\nConverted Predicted Price to other currencies:")
        target_currencies = ['USD', 'EUR', 'GBP', 'AUD', 'CAD', 'JPY']
        for currency in target_currencies:
            converted_price = convert_currency(price, rates, currency)
            if converted_price is not None:
                print(f"{currency}: {converted_price:.2f}")
else:
    predicted_price = predict_price(er_specifications)
    print(f"Predicted Price: ₹{predicted_price:.2f}")

    rates = get_exchange_rates("INR")
    if rates:
        print("\nConverted Predicted Price to other currencies:")
        target_currencies = ['USD', 'EUR', 'GBP', 'AUD', 'CAD', 'JPY']
        for currency in target_currencies:
            converted_price = convert_currency(predicted_price, rates, currency)
            if converted_price is not None:
                print(f"{currency}: {converted_price:.2f}")

CORS(app) 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        brand = data.get('brand')
        print(f"Received Brand: {brand}")

        # Prepare the input for prediction
        input_data = pd.DataFrame({'brand': [brand]})
        print(f"Input Data: {input_data}")

        # Make the prediction
        prediction = model2.predict(input_data)
        print(f"Prediction Output: {prediction}")

        # Format the response
        response = {
            'Gold (g)': prediction[0][0],
            'Aluminum (g)': prediction[0][1],
            'Silver (g)': prediction[0][2],
            'Carbon (g)': prediction[0][3],
            'Platinum (g)': prediction[0][4],
            'Nickel (g)': prediction[0][5],
            'Lithium (g)': prediction[0][6],
            'Estimated Price (INR)' : prediction[0][7]
        }
        print(response)
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

from flask import Flask, render_template, Response, request, jsonify
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
target_classes = ['cell phone', 'laptop']

# Load the classification model (for phones: brand & model)
class_model = models.resnet18(pretrained=False)  # Assuming you're using ResNet18
num_ftrs = class_model.fc.in_features
class_model.fc = nn.Linear(num_ftrs, 2)  # Assuming two classes: 'Apple iphone', 'vivo IQ Z6 lite'
class_model.load_state_dict(torch.load('laptop_classifier.pth'))
class_model.eval()

# Define the classes for the classification model
class_names = ['Apple iphone', 'vivo IQ Z6 lite']

# Define the video capture
cap = cv2.VideoCapture(0)

# Define transformations for classification model input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# To store the latest captured frame
latest_frame = None
detection_details = {}

def count_objects(results, class_names):
    counts = {class_name: 0 for class_name in class_names}
    for det in results.xyxy[0]:  # Results.xyxy contains [x1, y1, x2, y2, confidence, class]
        class_id = int(det[5])
        class_name = yolo_model.names[class_id]
        if class_name in class_names:
            counts[class_name] += 1
    return counts
import cv2
import torch
import torchvision.transforms as transforms
import time
from flask import Flask, render_template, Response, request, jsonify
from torchvision import models
import torch.nn as nn
import os

def generate_frames():
    global latest_frame, detection_details
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform object detection (stage 1)
        results = yolo_model(frame)
        results.render()  # Draw bounding boxes
        counts = count_objects(results, target_classes)

        # Update detection details for displaying in GUI
        detection_details = counts

        # Display object counts on the frame
        cv2.putText(frame, f"Cell Phones: {counts['cell phone']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Laptops: {counts['laptop']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # If a cell phone is detected, run classification model (stage 2)
        if counts['cell phone'] > 0:
            # Convert the frame to be compatible with the ResNet model
            image_tensor = transform(frame).unsqueeze(0)

            # Run the classification model
            with torch.no_grad():
                outputs = class_model(image_tensor)
                _, preds = torch.max(outputs, 1)
                predicted_class = class_names[preds[0]]

            # Update detection details with the classification result
            detection_details['classification'] = predicted_class

            # Draw the predicted class (brand/model) on the frame
            cv2.putText(frame, predicted_class, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Store the latest frame for image capture
        latest_frame = frame.copy()

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/capture_image', methods=['POST'])
def capture_image():
    if latest_frame is not None:
        # Save the latest frame as an image
        filename = os.path.join('static', 'captured_image.jpg')
        cv2.imwrite(filename, latest_frame)
        return jsonify({'message': 'Image captured successfully!', 'image_url': filename, 'details': detection_details})
    else:
        return jsonify({'message': 'No frame available to capture!'})

if __name__ == '__main__':
    app.run(debug=True)
