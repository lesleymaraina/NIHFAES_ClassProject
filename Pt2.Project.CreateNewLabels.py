################################################################
#About
#DATE MAY 22 2017
#Purpose: generate new labels based on predict prbability
################################################################


'''
Imports
'''
import pandas as pd
import numpy as np


'''
Code
'''

df = pd.read_csv('/Users/lmc2/tryX.csv')

'''
Store all of the datapoints in one column based on predict_proba
'''

def f(row):
    if row['one'] == 1:
        val = 'one1'
    elif row['one'] == 0.9:
        val = 'one0.9'
    elif row['one'] == 0.8:
        val = 'one0.8'
    elif row['one'] == 0.7:
        val = 'one0.7'
    elif row['one'] == 0.6:
        val = 'one0.6'
    elif row['two'] == 1:
        val = 'two1'
    elif row['two'] == 0.9:
        val = 'two0.9'
    elif row['two'] == 0.8:
        val = 'two0.8'
    elif row['two'] == 0.7:
        val = 'two0.7'
    elif row['two'] == 0.6:
        val = 'two0.6'
    elif row['three'] == 1:
        val = 'three1'
    elif row['three'] == 0.9:
        val = 'three0.9'
    elif row['three'] == 0.8:
        val = 'three0.8'
    elif row['three'] == 0.7:
        val = 'three0.7'
    elif row['three'] == 0.6:
        val = 'three0.6'
    else:
        val = -1
    return val

'''
Export newly labeled dataframe
'''

df['all_label'] = df.apply(f, axis=1)
df.to_csv('practice_label.csv', index=False)

