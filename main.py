from flask import Flask, render_template, request
from constants import dataArray
from algorithm import ANN_algo

app = Flask(__name__)

@app.route('/')
def hello_world():
   # print(dataArray)
   return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
   if request.method=="POST":
      dataArray_ = dataArray
      season = request.form.get("season")
      crop = request.form.get("crop")
      area = request.form.get("area")
      soilType = request.form.get('soil_type')
      district = request.form.get('district')

      dataArray_[season] = 1
      dataArray_[crop] = 1
      dataArray_[soilType] = 1
      dataArray_[district] = 1

      area = 453200
      temperature = 23.1
      precipitation = 8.7838
      humidity = 70

      l = [area, temperature, precipitation, humidity]
      for key,value in dataArray_.items():
         l.append(value)
      
      # print('>>>>>>>>>>>>>>>>>>>>>>>>>>> ',len(l), len(dataArray_))

      val = ANN_algo(l)
      print('>>>>>>>>>>>>>>>>>>>>>>>> val-', val[0][0])
      st = "<h1> Predicted {} </h1>".format(val[0][0])
      return st

if __name__ == '__main__':
   app.run(debug=True)