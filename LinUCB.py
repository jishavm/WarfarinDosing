import numpy as np
import time
import random
import dataPreprocessing_test
from dataPreprocessing_test import dosage_func
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

ALPHA = 0.2
K = 3




class LinUCB():
	def __init__(self, feature_length):
		self.feature_length = feature_length
		self.alpha = ALPHA
		self.A = [np.identity(self.feature_length +1) for k in range(K)]
		self.b = [np.zeros(self.feature_length+1) for k in range(K)]
		self.A_inv = [np.identity(self.feature_length+1) for k in range(K)]
		self.theta = [np.zeros(self.feature_length+1) for k in range(K)]
		self.cummulative_regret = 0
		self.regret = []
		self.DOSAGE=['low','medium','high']



	def getDose(self, sample, correct_dose):
		x_t = sample
		x = np.transpose(x_t)
		
		p = np.zeros(K)

		for i in range(len(self.A)):
			
			
			a = sample.T.dot(self.A_inv[i])
			
			b = self.theta[i].dot(sample)
			
			c = self.alpha * np.sqrt(a.dot(sample))
			p[i] = b + c
			
			
		a_max = np.argmax(p)
		
		dose = self.DOSAGE[a_max]

		if dose == dosage_func(correct_dose):
			r = 0
		else:
			r= -1

		self.A[a_max] += np.matmul(x, x_t)
		self.b[a_max] += r * x
		self.A_inv[a_max] = np.linalg.inv(self.A[a_max])
		self.theta[a_max] = self.A_inv[a_max].dot(self.b[a_max])
		

		return dose


	def train(self, data):
		#data can be dataframe
		d = len(data.index) 
		df = data.sample(frac=1).reset_index(drop=True)
		#df=data
		
		X = df.drop(['Therapeutic_Dose_of_Warfarin'], axis=1)
		X['bias'] = 1
		X = X.as_matrix()

		total_sample = 0
		correct_sample =0

		for i in range(d + 1):
			
			if (i == d):
				print("end")
				break
			sample = X[i]
			correct_dose = df['Therapeutic_Dose_of_Warfarin'].iloc[i]
			predicted_dose = self.getDose(sample, correct_dose)
			if predicted_dose == dosage_func(correct_dose):
				r = 0
			else:
				r= -1
				self.cummulative_regret +=1
			if r >=0:
				correct_sample +=1
			total_sample+=1	
			self.regret.append(self.cummulative_regret)

		print(correct_sample /total_sample)		

		
def main():
	data = dataPreprocessing_test.preprocess()

	
	linUCB = LinUCB(len(data.columns) - 1) # passing the number of features
	linUCB.train(data)
	
	x = list(range(len(data.index)))
	y = linUCB.regret
	plt.title("Cumulative Regret of Linear UCB(High Dimensional Space)")
	plt.xlabel("t")
	plt.ylabel("cumulative Regret @ t");
	plt.plot(x, x, linestyle = '--')
	plt.plot(x, y, linestyle = '-', label = 'LinUCB')
	plt.legend();
	plt.savefig('/Users/muthiyil/Documents/CS234/default_project/WarfarinDosing/plots/HighDimlinUCB.png')


if __name__ == "__main__":
	main()













