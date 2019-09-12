import numpy as np
import pandas as pd
import dataPreprocessingBaselines
from dataPreprocessingBaselines import dosage_func  
def accu(actual,pred):
    #
    #Calculating the accuracy of the method, percentage of correct predictions
    #
        res = []
        for i in range(len(actual)):
            if actual[i] == pred[i]:
                res.append(1)
            else:
                res.append(0)
            
        return np.sum(res)/len(res)


def clinical_dosage_algorithm(df_clin):
   


    df_clin['clinical_dosage'] = ((4.0376 
                            - ( 0.2546 * df_clin['age_in_decades']) 
                            + ( 0.0118 * df_clin['Height (cm)'] ) 
                            + ( 0.0134 * df_clin['Weight (kg)'])
                            - ( 0.6752 * df_clin['is_asian']) 
                            + ( 0.4060 * df_clin['is_black'])
                            + ( 0.0443 * df_clin['is_Missing_or_Mixed_race'])  
                            + ( 1.2799 * df_clin['enzyme_inducer_status'])
                            - ( 0.5695 * df_clin['Amiodarone_status'])      
                            ) **2 )

    df_clin['right_clinical_dosage'] = df_clin['clinical_dosage'].apply(dosage_func)   
    
    return (accu(df_clin['right_dosage'],df_clin['right_clinical_dosage']))

def fixed_dosage(fixed_dataframe):
    
    df_new = fixed_dataframe[['Therapeutic Dose of Warfarin','right_dosage']]
    df_new['fixed_dosage'] = 35
    df_new['right_dosage_fixed'] = df_new['fixed_dosage'].apply(dosage_func)
    return (accu(df_new['right_dosage'],df_new['right_dosage_fixed']))


def main():
    data = dataPreprocessingBaselines.preprocess()
    df_frac = data.sample(frac=0.2)
    df_frac.reset_index(inplace=True)
    print(fixed_dosage(df_frac))
    print(clinical_dosage_algorithm(df_frac))


if __name__ == "__main__":
	main()    

