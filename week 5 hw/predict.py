#!/usr/bin/env python
# coding: utf-8
# In[3]:
import pickle
import numpy as np
from flask import Flask, request, jsonify
import sklearn


def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

# In[15]: open and load the model from the directory
with open ('dv.bin', 'rb' ) as f_in:
    dv = pickle.load(f_in)
with open ('model1.bin', 'rb' ) as f_in:
    model = pickle.load(f_in)



# In[17]:
app = Flask('churn')

@app.route('/predict', methods=['Post'])
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5
    
    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
