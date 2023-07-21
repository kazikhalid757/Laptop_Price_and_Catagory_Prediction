import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from flask import Flask, render_template, request
import streamlit as st

# Read the CSV file
df = pd.read_csv("Cleaned_Laptop_data_with_category.csv")



# Convert "processor_name" column to string data type
df["processor_name"] = df["processor_name"].astype(str)

# Feature Selection: Select relevant features for prediction
X = df[['ram_gb', 'ssd', 'processor_name', 'graphic_card_gb']]
y_category = df['Laptop_Category']
y_price = df['latest_price']

# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['processor_name'])

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_category_train, y_category_test, y_price_train, y_price_test = train_test_split(
    X, y_category, y_price, test_size=0.2, random_state=42)

# Train the Laptop Category Prediction Model (Random Forest Classifier)
category_model = RandomForestClassifier(n_estimators=100, random_state=42)
category_model.fit(X_train, y_category_train)

# Train the Price Prediction Model (Linear Regression)
price_model = LinearRegression()
price_model.fit(X_train, y_price_train)

# Predict the Laptop Category and Price for the test data
y_category_pred = category_model.predict(X_test)
y_price_pred = price_model.predict(X_test)

# Evaluate the Laptop Category Prediction Model
category_accuracy = accuracy_score(y_category_test, y_category_pred)
print("Laptop Category Prediction Accuracy:", category_accuracy)

# Evaluate the Price Prediction Model
price_rmse = mean_squared_error(y_price_test, y_price_pred, squared=False)
print("Price Prediction Root Mean Squared Error (RMSE):", price_rmse)

# Function to input new data and get predictions
def predict_laptop_category_and_price(new_data):
    # Create a DataFrame with all the columns used during training
    new_data = pd.get_dummies(new_data, columns=['processor_name'])
    new_data = new_data.reindex(columns=X.columns, fill_value=0)  # Fill missing columns with 0
    
    # Predict Laptop Category
    laptop_category_prediction = category_model.predict(new_data)
    
    # Predict Price
    laptop_price_prediction = price_model.predict(new_data)
    
    return laptop_category_prediction[0], laptop_price_prediction[0]

# Input new data as a dictionary


# Define the main function to create the Streamlit app
def main():
    st.title('Laptop Category and Price Prediction')

    # Create input form using Streamlit widgets
    ram_gb = st.number_input('RAM (GB)', min_value=0, max_value=64, value=8)
    ssd = st.number_input('SSD (GB)', min_value=0, value=256)
    processor_name = st.text_input('Processor Name (Core i7 / Ryzen 7)', value='Core i7')
    graphic_card_gb = st.number_input('Graphic Card (GB)', min_value=0, value=2)

    if st.button('Predict'):
        # Create a DataFrame from the user input
        new_data = {
            'ram_gb': [ram_gb],
            'ssd': [ssd],
            'processor_name': [processor_name],
            'graphic_card_gb': [graphic_card_gb]
        }
        new_data_df = pd.DataFrame(new_data)

        # Get predictions for the new data
        predicted_category, predicted_price = predict_laptop_category_and_price(new_data_df)

        # Display the predicted values
        st.subheader('Predicted Laptop Category:')
        st.write(predicted_category)

        st.subheader('Predicted Laptop Price:')
        st.write(f'TK {predicted_price:.2f}')

# Run the Streamlit app
if __name__ == '__main__':
    main()


    
    
    

