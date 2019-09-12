import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def dosage_func(x):
    #
    #Method to set the categorical dosage value
    #
    if x < 21:
        return  'low'
    elif x >= 21 and x <=49:
        return  'medium'
    else:
        return 'high' 

def amiodarone_status(df):
	if ( df.get_value('Amiodarone (Cordarone)') > 0) :
		return 1
	else:
		return 0 
def enzyme_inducer_status(df):
	if ( (df.get_value('Carbamazepine (Tegretol)') > 0) and
		(df.get_value('Phenytoin (Dilantin)') > 0) and
		(df.get_value('Rifampin or Rifampicin') > 0) ):
		return 1
	else:
		return 0
def age_in_decades(x):    
    if x == '10 - 19':
        return 1
    elif x == '20 - 29':
        return 2
    elif x == '30 - 39':
        return 3
    elif x == '40 - 49':
        return 4
    elif x == '50 - 59':
        return 5
    elif x == '60 - 69':
        return 6
    elif x == '70 - 79':
        return 7
    elif x == '80 - 89':
        return 8
    elif x == '90+':
        return 9
    else:
        return 0


def preprocess():
	df = pd.read_csv("/Users/muthiyil/Documents/stanford/project/data 2/warfarin.csv")
	df['right_dosage'] = df['Therapeutic Dose of Warfarin'].apply(dosage_func)
	df = df[['Age','Race','Carbamazepine (Tegretol)','Phenytoin (Dilantin)','Rifampin or Rifampicin','Amiodarone (Cordarone)',
	'Height (cm)','Weight (kg)','right_dosage','Therapeutic Dose of Warfarin']]
	df.dropna(how='any', inplace= True)
	df.reset_index(inplace=True)
	df['age_in_decades'] = df['Age'].apply(age_in_decades)
	df['is_asian'] = df.Race.apply(lambda x: 1 if x == 'Asian' else 0)
	df['is_black'] = df.Race.apply(lambda x: 1 if x == 'Black or African American' else 0)
	df['is_Missing_or_Mixed_race']= df.Race.apply(lambda x: 1 if x == 'Unknown' else 0)
	df['enzyme_inducer_status'] = df.apply(enzyme_inducer_status,axis=1)
	df['Amiodarone_status'] = df.apply(amiodarone_status,axis=1)

	#print(df.columns.values)
	return df




if __name__ == "__main__":
	preprocess()



