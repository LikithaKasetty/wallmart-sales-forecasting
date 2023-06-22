from flask import Flask, render_template, request
import pickle
#Train the model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#Read the csv file
df = pd.read_csv(r'C:\Users\Likitha Kasetty\OneDrive\Desktop\Walmart_sales_prediction\Walmart Data Analysis and Forcasting.csv')
df['Revenue']=df.apply(lambda x :float(x['Weekly_Sales'])-float(x['CPI'])-float(x['Fuel_Price']),axis= 1 )
x=df[['CPI','Revenue', 'Store', 'Fuel_Price']]
y=df['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=44, shuffle =True)
LinearRegressionModel = LinearRegression(fit_intercept=True,copy_X=True,n_jobs=-1)
LinearRegressionModel.fit(X_train, y_train)
#Calculating Prediction
y_pred = LinearRegressionModel.predict(X_test)
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') 
print('Mean Absolute Error Value is : ', MAEValue)
#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') 
print('Mean Squared Error Value is : ', MSEValue)
#save the model
file = open("weeklysales_model.pkl", 'wb')
pickle.dump(LinearRegressionModel, file)
file.close()

#Create a web app instance
app = Flask(__name__)
model = pickle.load(open('weeklysales_model.pkl', 'rb')) #read mode
@app.route('/')
def home():
    return render_template('prediction.html')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        feature_1 = float(request.form['feature1'])
        feature_2 = float(request.form['feature2'])
        feature_3 = int(request.form['feature3'])
        feature_4 = float(request.form['feature4'])

        #get prediction
        input_cols = [[feature_1, feature_2, feature_3, feature_4]]
        prediction = model.predict(input_cols)
        output = round(prediction[0], 2)
        #output to HTML page
        return render_template("prediction.html", prediction_text='Your predicted weekly sales of wallmart is $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)  

#GET AND POST request : HTTP request
