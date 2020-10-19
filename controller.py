from flask import Flask, request, render_template
import model as m
import test as t

app = Flask(__name__)

@app.route("/", methods=['POST','GET'])
def home():
	if request.method == 'GET':
		return render_template("index.html")
	else:
		# var_list = ['Pclass','Sex','Age_band','Title','Embarked','Fare_cat','Fare_cat','Alone','Family_Size','SibSp','Parch']
		var_list = ['Pclass','Sex','Age_band','Title','Embarked','Fare_cat','SibSp','Parch']
		data = {}
		for var in var_list:
			data[var] = int(request.form[var])
		
		result = t.prediction(data)
		return render_template("index.html", result=result)

if __name__ == '__main__':
	app.run(debug=True)