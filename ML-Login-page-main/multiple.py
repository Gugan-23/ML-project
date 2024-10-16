import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

from forex_python.converter import CurrencyRates
import numpy as np
from flask import Flask, request, jsonify
import requests
app = Flask(__name__)

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
print(data['Product'].unique())
X = data.drop(['newprice', 'price', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1, errors='ignore')
y = data['price']

print("Columns in the DataFrame:", data.columns)

unique_brands = data['brand'].unique()

print("Unique Brands:")
for brand in unique_brands:
    print(brand)

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

print(X.isnull().sum())  

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
        
    
    if display_size < 7:
        return "Phone"
    elif 7 <= display_size < 15:
        return "Phablet"  # Optional: for larger phones
    elif 15 <= display_size < 20:
        return "Laptop"
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

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

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

import tkinter as tk

# Initial specifications dictionary
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
    'display_size': 7,
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

# Function to update specifications
def display_specifications():
    # Clear previous labels
    for widget in display_frame.winfo_children():
        widget.destroy()

    for spec, value in er_specifications.items():
        label = tk.Label(display_frame, text=f"{spec.replace('_', ' ').title()}: {value}")
        label.pack(pady=2)

# Function to update specified features
def update_specifications():
    # Update only name, brand, and display size
    er_specifications['brand'] = brand_entry.get() or er_specifications['brand']  # Use existing value if empty
    er_specifications['name'] = name_entry.get() or er_specifications['name']
    try:
        display_size = int(display_size_entry.get())
        er_specifications['display_size'] = display_size
    except ValueError:
        pass  # If input is invalid, keep existing value

    display_specifications()  # Refresh the displayed specifications

# Create the main window
root = tk.Tk()
root.title("Device Specifications")
root.geometry("400x400")

# Create a frame for the entries
entry_frame = tk.Frame(root)
entry_frame.pack(pady=10)

# Create entry fields for the selected specifications
tk.Label(entry_frame, text="Brand:").pack(anchor="w")
brand_entry = tk.Entry(entry_frame)
brand_entry.pack(fill="x", padx=10, pady=5)

tk.Label(entry_frame, text="Name:").pack(anchor="w")
name_entry = tk.Entry(entry_frame)
name_entry.pack(fill="x", padx=10, pady=5)

tk.Label(entry_frame, text="Display Size:").pack(anchor="w")
display_size_entry = tk.Entry(entry_frame)
display_size_entry.pack(fill="x", padx=10, pady=5)

# Create a button to update specifications
update_button = tk.Button(entry_frame, text="Update Specifications", command=update_specifications)
update_button.pack(pady=10)

# Create a frame for the scrollbar and canvas
scrollable_frame = tk.Frame(root)
scrollable_frame.pack(pady=10, fill="both", expand=True)

# Create a canvas for scrolling
canvas = tk.Canvas(scrollable_frame)
scrollbar = tk.Scrollbar(scrollable_frame, orient="vertical", command=canvas.yview)
display_frame = tk.Frame(canvas)

# Configure the canvas
canvas.create_window((0, 0), window=display_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Pack the canvas and scrollbar
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Bind the configure event to update the scrollbar
def configure_scroll_region(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

display_frame.bind("<Configure>", configure_scroll_region)

# Initial display of specifications
display_specifications()

# Run the application
root.mainloop()

# Call the function with user_input
price = find_price(er_specifications, data)


def get_brand_details(brand_name):
    # Load the dataset
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
