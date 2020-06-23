from flask import Flask,render_template,request,redirect,url_for
import joblib
import numpy as np
app=Flask(__name__)
model=joblib.load('linear_model.h5')




@app.route("/",methods=['POST','GET'])
def home():
    return render_template('home.html')
    

@app.route("/pre",methods=['GET','POST'])
def Predict():
    if request.method=='POST':
        California=float(request.form['California'])
        Florida=float(request.form['Florida'])
        NewYork=float(request.form['NewYork'])
        R_n_D=float(request.form['R_n_D'])
        Administration=float(request.form['Administration'])
        Marketing=float(request.form['Marketing'])
        X=np.array([California,Florida,NewYork,R_n_D,Administration,Marketing])
        X=X.reshape(1,-1)
        Prediction=model.predict(X)
        Prediction=round(float(Prediction),2)
    return render_template('Prediction.html',value=Prediction)
if __name__=="__main":
    app.run(debug=True)
