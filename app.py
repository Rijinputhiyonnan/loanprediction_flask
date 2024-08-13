from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

df =pd.read_csv("train_ctrUa4K.csv")

with open('model_loanpred.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
    
@app.route('/')

def home():
    return render_template('home.html')


from sklearn.preprocessing import LabelEncoder

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form["gender"]
    married = request.form["married"]
    dependents = request.form["dependents"]
    education = request.form["education"]
    self_employed = request.form["self_employed"]
    applicant_income = request.form["applicant_income"]
    coapplicant_income = request.form["coapplicant_income"]
    loan_amount = request.form["loan_amount"]
    loan_amount_term = request.form["loan_amount_term"]
    credit_history = request.form["credit_history"]
    property_area = request.form["property_area"]
    
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })
    
    input_data = pd.get_dummies(input_data, dtype='int64')
    
    # List of expected features based on training data
    expected_features = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
        'Gender_Female', 'Gender_Male', 'Married_No', 'Married_Yes', 'Dependents_0', 'Dependents_1',
        'Dependents_2', 'Dependents_3+', 'Education_Graduate', 'Education_Not Graduate',
        'Self_Employed_No', 'Self_Employed_Yes', 'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban'
    ]
    
    # Add missing columns with default value 0
    for feature in expected_features:
        if feature not in input_data.columns:
            input_data[feature] = 0
    
    # Ensure the columns are in the same order as expected
    input_data = input_data[expected_features]
    
    print(input_data, "values")
        
    pred = model.predict(input_data)
    
    # Assuming you have a saved LabelEncoder, otherwise, you need to fit it again
    le = LabelEncoder()
    # le.fit(your_original_labels)  # Fit it with your original labels if not saved
    
    #prediction = le.inverse_transform(pred)
    print(pred, "prediction")
    if pred[0] > 0.5:
        ninja = "Eligible for Loan"
    else:
        ninja = "Not eligible"
        
    return render_template("home.html", p_result=ninja)



if __name__ == '__main__':
    app.run(debug =True)
    