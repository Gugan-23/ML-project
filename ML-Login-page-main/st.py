import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

def run_random_forest_model(file_path):
    # Load data from Excel file
    data = pd.read_excel(file_path)

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
    print(f"R² Score: {r2:.4f}")

# Call the function with the path to your Excel file
run_random_forest_model(r'C:\Users\Administrator\Downloads\archive (2)\data1.xlsx')
