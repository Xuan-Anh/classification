# -*- coding: utf-8 -*-
"""APIVcc.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/184BBiDfIRH98a6SIteOeOaNoR_H5tyne

## Phần 1
Xử lý dữ liệu đầu vào

###Load du lieu
"""

from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim # thư viện NLP
content = "day la tro choi"
X_test = [content]
def xuLy(X_data_all):
  import numpy as np
  X_data_ok = np.array(X_data_all)
  X_data_ok.astype('str')
  X_data = []
  import pandas as pd
  import gensim
  for line in X_data_ok:
    line = gensim.utils.simple_preprocess(line) 
    line_1 = ' '.join(line)
    line_2 = ViTokenizer.tokenize(line_1)
    X_data.append(line_2)
  return X_data

def classify(X_test):
  X_test = xuLy(X_test) 
  import pickle

  # tfidf thường
  path = open("Game_tfidf.pkl", 'rb')
  tfidf_vect = pickle.load(path)
  X_test_tfidf =  tfidf_vect.transform(X_test)
  # tfidf_svd
  from sklearn.decomposition import TruncatedSVD

  svd = TruncatedSVD(n_components=300, random_state=42)
  path = open("Game_svd.pkl", 'rb')
  X_data_tfidf = pickle.load(path)
  svd.fit(X_data_tfidf)
  X_test_tfidf_svd = svd.transform(X_test_tfidf)
  import pickle
  path = open("Game_mode.pkl", 'rb')
  model = pickle.load(path)
  from sklearn import metrics
  test_predictions = model.predict(X_test_tfidf_svd)
  print(test_predictions)
  # import json
  # result = json.dumps(test_predictions.tolist())
  # return result
  return test_predictions



# from flask_ngrok import run_with_ngrok
from flask import Flask
from flask import jsonify
import json
app = Flask(__name__)
# run_with_ngrok(app)   #starts ngrok when the app is run

@app.route("/ping")
def test(request):
  return json({"hello": "world"})



# @app.route('/news_classify', methods=['GET'])
# def classification_content(request):
#     title = request.args.get("title")
#     descriptions = request.args.get("descriptions")
#     content = request.args.get("content")
#     return 
@app.route("/", methods = ['GET'])
def home():
  return jsonify(classify(X_test).tolist())
if __name__ == '__main__':
    # app.run()
  app.run()