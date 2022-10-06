from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
with open("model.pickle","rb") as f:
    model=pickle.load(f)
with open("std.pickle","rb") as f:
    std=pickle.load(f)
@app.route('/',methods=["GET"])
def fun1():
    return render_template("index.html")
@app.route("/rent",methods=["POST"])
def pred():
    gen=float(request.form['g'])
    age=float(request.form['age'])
    sal=float(request.form['sal'])
   

    data=np.array([[gen,age,sal]])
    trans_data=std.transform(data)
    result=model.predict(trans_data)
    return render_template('index.html',prediction=result)
if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080,debug=False)
