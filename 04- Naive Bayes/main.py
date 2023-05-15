import numpy as np
import pandas as pd
import math
from model import NaiveBayes

def pre_processing(df):

	X = df.drop([df.columns[-1]], axis = 1)
	y = df[df.columns[-1]]

	return X, y



if __name__ == "__main__":



	df = pd.read_table("weather.txt")


	X,Y  = pre_processing(df)

	model = NaiveBayes()
	model.fit(X,Y)
	print(model.predict(X))






