import numpy as np
import pickle
import pandas as pd
import sys
import ast
from flask import Flask, request, jsonify, render_template


pickle_in = open("model.pkl","rb")
random = pickle.load(pickle_in)

"""
For rendering results on HTML GUI
"""
int_features = ast.literal_eval(sys.argv[1])
final_features = [np.array(int_features)]
prediction = random.predict(final_features)
print(str(prediction[0]))
   