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
# Load the trained model
model = joblib.load('brand_to_elements_model.pkl')

app = Flask(__name__)
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
@app.route('/predict', methods=['POST'])
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
        prediction = model.predict(input_data)
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

    except Exception as e:
        return jsonify({'error': str(e)}), 500
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

if __name__ == '__main__':
    app.run(debug=True)
