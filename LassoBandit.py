import numpy as np
import time
import random
import dataPreprocessing_test
from dataPreprocessing_test import dosage_func
from sklearn import linear_model
import math
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

K = 3

class LassoBandit:
	def __init__(self):
		self.q = 1 
		self.h = 25
		self.l1 = 0.05
		self.l2 = 0.05


		self.tau = []
		self.forced_est = [None for i in range(K)]
		self.forced_samples = [[] for i in range(K) ]
		self.forced_sample_rewards = [[] for i in range(K)]
		
		
		self.all_sample_est = [None for i in range(K)]
		self.all_samples =[[] for i in range(K)]
		self.all_sample_rewards = [[] for i in range(K)]

		self.DOSAGE=['low','medium','high']
		self.cummulative_regret = 0
		self.regret = []

		

	

	def getDose_original(self, sample, correct_dose, t):
		d = (len(sample))
		pi_t = None
		forced_sampling = False
		for i in range(len(self.tau)):
			print("checking forced samples................for t={:d}".format(t))
			if(t in self.tau[i]):
				forced_sampling = True
				pi_t = i
				break
		print("action for t={} is {}".format(t,pi_t))
		
		if (not forced_sampling):
			forced_ests = []
			for beta in self.forced_est:
				forced_ests.append(beta.predict([sample.tolist()])[0])
			boundary = max(forced_ests) - self.h /2

			a_max = None
			val_max = None
			for a in range(K):
				if forced_ests[a] < boundary:
					continue
				val = float(self.all_sample_est[a].predict([sample.tolist()])[0]	)
				if(a_max is None or val > val_max):
					a_max = a
					val_max = val
				pi_t = a_max
		
		dose = self.DOSAGE[pi_t]
		if dose == dosage_func(correct_dose):
			r = 0
		else:
			r= -1

		
		#Updating forced_estimators
		if (forced_sampling):
			self.forced_samples[pi_t].append(sample)
			self.forced_sample_rewards[pi_t].append(r)
			lasso = linear_model.Lasso(alpha=self.l1, fit_intercept=True)
			lasso.fit(self.forced_samples[pi_t], self.forced_sample_rewards[pi_t])
			self.forced_est[pi_t] = lasso
		
		self.all_samples[pi_t].append(sample)
		self.all_sample_rewards[pi_t].append(r)
		self.l2 = self.l1 * math.sqrt((math.log(t) +math.log(d))/t)
		
		for j in range(K):
			if(len(self.all_samples[j]) == 0):
				continue
			lasso = linear_model.Lasso(alpha = self.l2, fit_intercept = True)
			lasso.fit(self.all_samples[j], self.all_sample_rewards[j])
			self.all_sample_est[j] = lasso

		return dose		

	

	def train(self,data):
		df = data.sample(frac=1).reset_index(drop=True)
		d = len(data.index)
		
		X = df.drop(['Therapeutic_Dose_of_Warfarin'], axis=1)
		X = X.as_matrix()
		
		#Tau initialization 
		for i in range(1, (K+1)):
			_tau = []
			n = 0
			run = True
			while(run):
				for j in range(self.q*(i-1)+1, self.q*i+1):
					forced_index = (math.pow(2,n)-1) * K * self.q + j
					if (forced_index <10000):
						_tau.append(forced_index)
					else:
						run = False
						break
				n +=1							
			self.tau.append(set(_tau))


		total_sample = 0
		correct_sample =0

		for i in range(d+1):
			if i==d:
				print("----end-----")
				break
			sample = X[i]
			correct_dose = df['Therapeutic_Dose_of_Warfarin'].iloc[i]
			# print("\n before calling getDose")
			# print("sample:")
			# print(sample)
			# print("t: {:d}".format(i+1))
			predicted_dose = self.getDose_original((sample), correct_dose, i+1)
			print("predicted Dose = ",predicted_dose)
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
	

	lassobandit = LassoBandit()
	lassobandit.train(data)
	x = list(range(len(data.index)))
	y = lassobandit.regret	
	plt.title("Cumulative Regret of Lasso Bandit - High Dim")
	plt.xlabel("t")
	plt.ylabel("cumulative regret @ t");
	plt.plot(x, x, linestyle = '--')
	plt.plot(x, y, linestyle = '-', label = 'lassoBandit')
	plt.legend();
	plt.savefig('HighDimlasso.png')


if __name__=="__main__":
	main()