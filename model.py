import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier

#Data Wrangling
class Train_Model:

	def __init__(self,data):
		self.data = data

	def train(self):
		self.data['Title']=self.data.Name.str.extract(r'([A-Za-z]+)\.') #lets extract the Salutations
		self.data['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

		## Assigning the NaN Values with the Ceil values of the mean ages
		self.data.loc[(self.data.Age.isnull())&(self.data.Initial=='Mr'),'Age']=33
		self.data.loc[(self.data.Age.isnull())&(self.data.Initial=='Mrs'),'Age']=36
		self.data.loc[(self.data.Age.isnull())&(self.data.Initial=='Master'),'Age']=5
		self.data.loc[(self.data.Age.isnull())&(self.data.Initial=='Miss'),'Age']=22
		self.data.loc[(self.data.Age.isnull())&(self.data.Initial=='Other'),'Age']=46

		self.data['Embarked'].fillna('S',inplace=True)

		self.data['Age_band']=0
		self.data.loc[self.data['Age']<=16,'Age_band']=0
		self.data.loc[(self.data['Age']>16)&(self.data['Age']<=32),'Age_band']=1
		self.data.loc[(self.data['Age']>32)&(self.data['Age']<=48),'Age_band']=2
		self.data.loc[(self.data['Age']>48)&(self.data['Age']<=64),'Age_band']=3
		self.data.loc[self.data['Age']>64,'Age_band']=4

		self.data['Family_Size']=0
		self.data['Family_Size']=self.data['Parch']+self.data['SibSp']#family size
		self.data['Alone']=0
		self.data.loc[self.data.Family_Size==0,'Alone']=1#Alone

		self.data['Fare_cat']=0
		self.data.loc[self.data['Fare']<=7.91,'Fare_cat']=0
		self.data.loc[(self.data['Fare']>7.91)&(self.data['Fare']<=14.454),'Fare_cat']=1
		self.data.loc[(self.data['Fare']>14.454)&(self.data['Fare']<=31),'Fare_cat']=2
		self.data.loc[(self.data['Fare']>31)&(self.data['Fare']<=513),'Fare_cat']=3

		self.data['Fare_Range']=pd.qcut(self.data['Fare'],4)

		self.data['Sex'].replace(['male','female'],[0,1],inplace=True)
		self.data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
		self.data['Title'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

		self.data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)

		self.X = self.data.iloc[:,1:]
		self.y = self.data['Survived']

	def model(self):
		self.model=DecisionTreeClassifier()
		self.model.fit(self.X,self.y)

		with open("model_data.pkl",'wb') as f:
			pickle.dump(self.model,f)

if __name__ == '__main__':
	data = pd.read_csv('raw_data/train.csv')
	train_model = Train_Model(data)
	train_model.train()
	train_model.model()