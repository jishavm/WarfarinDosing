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

def weight_buckets(x):    
    if x < 50 :
        return 'less_than_50'
    elif x >= 50 and x < 60:
        return 'between_50_and_60'
    elif x >= 60 and x < 70:
        return 'between 60 and 70'

    elif x >= 70 and x < 80:
        return 'between 70 and 80'
    elif x >= 80 and x < 90:
        return 'between 80 and 90'    
    elif x >= 90 and x < 100:
        return 'between 90 and 100'            
    else:
        return 'Greater than 100'       



def get_warfain_ind_dummies(df_wind):
    df_wind['Indication_for_Warfarin_Treatment'].fillna('unknown', inplace=True)
    df_wind['warfain_ind_1'] = df_wind.Indication_for_Warfarin_Treatment.apply(lambda x : any((i for i in x if i.find('1') >= 0))).astype(int)
    df_wind['warfain_ind_2'] = df_wind.Indication_for_Warfarin_Treatment.apply(lambda x : any((i for i in x if i.find('2') >= 0))).astype(int)
    df_wind['warfain_ind_3'] = df_wind.Indication_for_Warfarin_Treatment.apply(lambda x : any((i for i in x if i.find('3') >= 0))).astype(int)
    df_wind['warfain_ind_4'] = df_wind.Indication_for_Warfarin_Treatment.apply(lambda x : any((i for i in x if i.find('4') >= 0))).astype(int)
    df_wind['warfain_ind_5'] = df_wind.Indication_for_Warfarin_Treatment.apply(lambda x : any((i for i in x if i.find('5') >= 0))).astype(int)
    df_wind['warfain_ind_6'] = df_wind.Indication_for_Warfarin_Treatment.apply(lambda x : any((i for i in x if i.find('6') >= 0))).astype(int)
    df_wind['warfain_ind_7'] = df_wind.Indication_for_Warfarin_Treatment.apply(lambda x : any((i for i in x if i.find('7') >= 0))).astype(int)
    df_wind['warfain_ind_8'] = df_wind.Indication_for_Warfarin_Treatment.apply(lambda x : any((i for i in x if (i.find('8') >= 0 or i.find('unknown') >= 0)))).astype(int)
    df_wind_dummy = df_wind[['warfain_ind_1','warfain_ind_2','warfain_ind_3','warfain_ind_4','warfain_ind_5','warfain_ind_6','warfain_ind_7','warfain_ind_8']]
    df_wind_dummy[['warfain_ind_1','warfain_ind_2','warfain_ind_3','warfain_ind_4','warfain_ind_5','warfain_ind_6','warfain_ind_7','warfain_ind_8']] = df_wind_dummy[['warfain_ind_1','warfain_ind_2','warfain_ind_3',
                             'warfain_ind_4','warfain_ind_5','warfain_ind_6',
                            'warfain_ind_7','warfain_ind_8']].astype(int)
    #print(df_wind_dummy.columns.values)
    return df_wind_dummy



def preprocess():
    df = pd.read_csv("/Users/muthiyil/Documents/stanford/project/data 2/warfarin.csv")
    df.columns = df.columns.str.replace(' ', '_')

    df['Gender'].fillna('Unknown', inplace=True)
    dummy_Gender = pd.get_dummies(df['Gender'],prefix='Gender')

    df['Race'].fillna('Unknown', inplace=True)
    dummmy_race = pd.get_dummies(df['Race'], prefix = 'Race')

    df['Ethnicity'].fillna('Unknown', inplace=True)
    dummmy_ethnicity = pd.get_dummies(df['Ethnicity'], prefix = 'Ethnicity')

    df['age_in_decades'] = df['Age'].apply(age_in_decades)
    age_dummies = pd.get_dummies(df['age_in_decades'], prefix = 'Age')

    df['weight_buckets'] = df['Weight_(kg)'].apply(weight_buckets)
    weight_dummies = pd.get_dummies(df['weight_buckets'], prefix = 'Weight') 

    warfain_ind_dummies = get_warfain_ind_dummies(df)

    df['Diabetes'].fillna('Unknown', inplace=True)
    dummmy_Diabetes = pd.get_dummies(df['Diabetes'], prefix = 'Diabetes')

    df['Congestive_Heart_Failure_and/or_Cardiomyopathy'].fillna('Unknown', inplace=True)
    dummmy_HeartFailure = pd.get_dummies(df['Congestive_Heart_Failure_and/or_Cardiomyopathy'], prefix = 'HeartFailure')


    df['Valve_Replacement'].fillna('Unknown', inplace=True)
    dummmy_Valve_Replacement = pd.get_dummies(df['Valve_Replacement'], prefix = 'Valve_Replacement')

    df['Aspirin'].fillna('Unknown', inplace=True)
    dummmy_Aspirin = pd.get_dummies(df['Aspirin'], prefix = 'Aspirin')

    df['Acetaminophen_or_Paracetamol_(Tylenol)'].fillna('Unknown', inplace=True)
    dummmy_Tylenol = pd.get_dummies(df['Acetaminophen_or_Paracetamol_(Tylenol)'], prefix = 'Tylenol')

    df['Simvastatin_(Zocor)'].fillna('Unknown', inplace=True)
    dummmy_Zocor = pd.get_dummies(df['Simvastatin_(Zocor)'], prefix = 'Zocor')


    df['Atorvastatin_(Lipitor)'].fillna('Unknown', inplace=True)
    dummmy_Lipitor = pd.get_dummies(df['Atorvastatin_(Lipitor)'], prefix = 'Lipitor')

    df['Fluvastatin_(Lescol)'].fillna('Unknown', inplace=True)
    dummmy_Lescol = pd.get_dummies(df['Fluvastatin_(Lescol)'], prefix = 'Lescol')

    df['Lovastatin_(Mevacor)'].fillna('Unknown', inplace=True)
    dummmy_Mevacor = pd.get_dummies(df['Lovastatin_(Mevacor)'], prefix = 'Mevacor')

    df['Pravastatin_(Pravachol)'].fillna('Unknown', inplace=True)
    dummmy_Pravachol = pd.get_dummies(df['Pravastatin_(Pravachol)'], prefix = 'Pravachol')

    df['Rosuvastatin_(Crestor)'].fillna('Unknown', inplace=True)
    dummmy_Crestor = pd.get_dummies(df['Rosuvastatin_(Crestor)'], prefix = 'Crestor')

    df['Amiodarone_(Cordarone)'].fillna('Unknown', inplace=True)
    dummmy_Cordarone = pd.get_dummies(df['Amiodarone_(Cordarone)'], prefix = 'Cordarone')

    df['Cerivastatin_(Baycol)'].fillna('Unknown', inplace=True)
    dummmy_Baycol = pd.get_dummies(df['Cerivastatin_(Baycol)'], prefix = 'Baycol')
   

    df['Carbamazepine_(Tegretol)'].fillna('Unknown', inplace=True)
    dummmy_Tegretol = pd.get_dummies(df['Carbamazepine_(Tegretol)'], prefix = 'Tegretol')

    df['Phenytoin_(Dilantin)'].fillna('Unknown', inplace=True)
    dummmy_Dilantin = pd.get_dummies(df['Phenytoin_(Dilantin)'], prefix = 'Dilantin')


    df['Rifampin_or_Rifampicin'].fillna('Unknown', inplace=True)
    dummmy_Rifampin = pd.get_dummies(df['Rifampin_or_Rifampicin'], prefix = 'Rifampin')

    df['Sulfonamide_Antibiotics'].fillna('Unknown', inplace=True)
    dummmy_Sulfonamide_Antibiotics = pd.get_dummies(df['Sulfonamide_Antibiotics'], prefix = 'Sulfonamide_Antibiotics')

    df['Macrolide_Antibiotics'].fillna('Unknown', inplace=True)
    dummmy_Macrolide_Antibiotics = pd.get_dummies(df['Macrolide_Antibiotics'], prefix = 'Macrolide_Antibiotics')

    df['Anti-fungal_Azoles'].fillna('Unknown', inplace=True)
    dummmy_Azoles = pd.get_dummies(df['Anti-fungal_Azoles'], prefix = 'Azoles')

    df['Herbal_Medications,_Vitamins,_Supplements'].fillna('Unknown', inplace=True)
    dummmy_Herbal_Medications = pd.get_dummies(df['Herbal_Medications,_Vitamins,_Supplements'], prefix = 'Herbal_Medications')

    df['Current_Smoker'].fillna('unknown', inplace=True)
    dummy_smoker = pd.get_dummies(df['Current_Smoker'],prefix='Current_Smoker')

    df['Cyp2C9_genotypes'].fillna('Unknown', inplace=True)
    Cyp2C9_genotypes_dummies = pd.get_dummies(df['Cyp2C9_genotypes'],prefix='Cyp2C9')

    df['VKORC1_-1639_consensus'].fillna('Unknown', inplace=True)
    consensus1639_dummies = pd.get_dummies(df['VKORC1_-1639_consensus'],prefix='consensus-1639')

    df['VKORC1_497_consensus'].fillna('Unknown', inplace=True)
    consensus497_dummies = pd.get_dummies(df['VKORC1_497_consensus'],prefix='consensus497')

    df['VKORC1_1173_consensus'].fillna('Unknown', inplace=True)
    consensus1173_dummies = pd.get_dummies(df['VKORC1_1173_consensus'],prefix='consensus1173')

    df['VKORC1_1542_consensus'].fillna('Unknown', inplace=True)
    consensus1542_dummies = pd.get_dummies(df['VKORC1_1542_consensus'],prefix='consensus1542')

    df['VKORC1_3730_consensus'].fillna('Unknown', inplace=True)
    consensus3730_dummies = pd.get_dummies(df['VKORC1_3730_consensus'],prefix='consensus3730')

    df['VKORC1_2255_consensus'].fillna('Unknown', inplace=True)
    consensus2255_dummies = pd.get_dummies(df['VKORC1_2255_consensus'],prefix='consensus2255')

    df['VKORC1_-4451_consensus'].fillna('Unknown', inplace=True)
    consensus4451_dummies = pd.get_dummies(df['VKORC1_-4451_consensus'],prefix='consensus-4451')

    df['CYP2C9_consensus'].fillna('Unknown', inplace=True)
    consensusCYP2C9_dummies = pd.get_dummies(df['CYP2C9_consensus'],prefix='consensusCYP2C9')




    df_new = pd.concat([df,dummy_Gender,dummmy_race,dummmy_ethnicity,age_dummies,weight_dummies,warfain_ind_dummies,dummmy_Diabetes,dummmy_HeartFailure,dummmy_Valve_Replacement,\
        dummmy_Aspirin,dummmy_Tylenol,dummmy_Zocor,dummmy_Lipitor,dummmy_Lescol,dummmy_Mevacor,dummmy_Pravachol,dummmy_Crestor,dummmy_Cordarone,dummmy_Baycol,\
        dummmy_Tegretol,dummmy_Dilantin,dummmy_Rifampin,dummmy_Sulfonamide_Antibiotics,dummmy_Macrolide_Antibiotics,dummmy_Azoles,dummmy_Herbal_Medications,dummy_smoker,\
        Cyp2C9_genotypes_dummies,consensus1639_dummies,consensus497_dummies,consensus1173_dummies,consensus1542_dummies,consensus3730_dummies,consensus2255_dummies,consensus4451_dummies,\
        consensusCYP2C9_dummies
        ], axis=1)

    df_new.columns = df_new.columns.str.replace(' ', '_')
    

    

    features = [
        'warfain_ind_1','warfain_ind_2','warfain_ind_3','warfain_ind_4','warfain_ind_5','warfain_ind_6','warfain_ind_7','warfain_ind_8',
        'Gender_female','Gender_male',
        'Race_Asian','Race_Black_or_African_American','Race_White',
        'Ethnicity_Hispanic_or_Latino','Ethnicity_not_Hispanic_or_Latino',
        'Age_0','Age_1','Age_2','Age_3','Age_4','Age_5','Age_6','Age_7','Age_8','Age_9',
        'Weight_Greater_than_100','Weight_between_60_and_70','Weight_between_70_and_80','Weight_between_80_and_90',
        'Weight_between_90_and_100','Weight_between_50_and_60','Weight_less_than_50',
        'Diabetes_0.0','Diabetes_1.0',
        'HeartFailure_0.0','HeartFailure_1.0',
        'Valve_Replacement_0.0','Valve_Replacement_1.0',
        'Aspirin_0.0','Aspirin_1.0',
        'Tylenol_0.0','Tylenol_1.0',
        'Zocor_0.0','Zocor_1.0',
        'Lipitor_0.0','Lipitor_1.0',
        'Lescol_0.0','Lescol_1.0',
        'Mevacor_0.0','Mevacor_1.0',
        'Pravachol_0.0','Pravachol_1.0',
        'Crestor_0.0','Crestor_1.0',
        'Cordarone_0.0','Cordarone_1.0',
        'Baycol_0.0',
        'Tegretol_0.0','Tegretol_1.0',
        'Dilantin_0.0','Dilantin_1.0',
        'Rifampin_0.0','Rifampin_1.0',
        'Sulfonamide_Antibiotics_0.0','Sulfonamide_Antibiotics_1.0',
        'Macrolide_Antibiotics_0.0','Macrolide_Antibiotics_1.0',
        'Azoles_0.0','Azoles_1.0',
        'Herbal_Medications_0.0','Herbal_Medications_1.0',
        'Current_Smoker_0.0','Current_Smoker_1.0',
        'Cyp2C9_*1/*1','Cyp2C9_*1/*11','Cyp2C9_*1/*13',
        'Cyp2C9_*1/*14','Cyp2C9_*1/*2','Cyp2C9_*1/*3',
        'Cyp2C9_*1/*5','Cyp2C9_*1/*6','Cyp2C9_*2/*2',
        'Cyp2C9_*2/*3','Cyp2C9_*3/*3',
        'consensus-1639_A/A','consensus-1639_A/G', 'consensus-1639_G/G',
        'consensus497_G/G','consensus497_G/T','consensus497_T/T',
        'consensus1173_C/C','consensus1173_C/T','consensus1173_T/T',
        'consensus1542_C/C', 'consensus1542_C/G','consensus1542_G/G',
        'consensus3730_A/A','consensus3730_A/G','consensus3730_G/G',
        'consensus2255_C/C','consensus2255_C/T','consensus2255_T/T',
        'consensus-4451_A/A', 'consensus-4451_A/C','consensus-4451_C/C',
        'consensusCYP2C9_*1/*1', 'consensusCYP2C9_*1/*11','consensusCYP2C9_*1/*13',
        'consensusCYP2C9_*1/*14','consensusCYP2C9_*1/*2','consensusCYP2C9_*1/*3',
        'consensusCYP2C9_*1/*5','consensusCYP2C9_*1/*6','consensusCYP2C9_*2/*2',
        'consensusCYP2C9_*2/*3','consensusCYP2C9_*3/*3',
        'Therapeutic_Dose_of_Warfarin']
    
   


    df_new = df_new[features]

    #print(df_new.head(5))
    

    return df_new
	




if __name__ == "__main__":
	preprocess()



