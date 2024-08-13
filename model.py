import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



df = pd.read_csv("train_ctrUa4K.csv")
print(df.head(5))


df.drop('Loan_ID', axis =1, inplace = True)


x=df.drop('Loan_Status', axis =1)
y=df['Loan_Status']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
x = pd.get_dummies(x, dtype ='int64')


x['LoanAmount'].fillna(df['LoanAmount'].median(), inplace= True)
x['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace= True)
x['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


for i in x[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]:
  q1 = x[i].quantile(0.25)
  q3 = x[i].quantile(0.75)
  iqr = q3 - q1
  lower_bound = q1 - 1.5 * iqr
  upper_bound = q3 + 1.5 * iqr
  x[i] = np.where((x[i] < lower_bound) | (x[i] > upper_bound), x[i].median(), x[i])
  
  

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


from sklearn.linear_model import LinearRegression

lir = LinearRegression()  
lir.fit(x_train, y_train)

import pickle
with open("model_loanpred.pkl", "wb") as model_file:
    pickle.dump(lir, model_file)
