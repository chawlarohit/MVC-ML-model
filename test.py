import pickle
import pandas as pd


def prediction(data):
	with open("model_data.pkl",'rb') as f:
		model = pickle.load(f)

	data['Family_Size'] = data['Parch'] + data['SibSp']
	data['Alone'] = 0
	if data['Family_Size'] == 0:
		data['Alone'] = 1

	X = pd.DataFrame([data])

	print(X)

	y = model.predict(X)
	if y == 0:
		return "Not Survived"
	else:
		return "Survived"