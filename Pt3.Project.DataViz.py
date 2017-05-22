###########################################
# About
#DATE MAY 22 2017
#Purpose: generate tSNE plots for predicted labels
# Resources
# http://bokeh.pydata.org/en/0.11.0/docs/user_guide/charts.html
# https://www.youtube.com/watch?v=NhTRrnLHTTc
# Many examples
# http://pbpython.com/visualization-tools-1.html

###########################################

###########################################
# Imports
###########################################
import pandas as pd
import numpy as np
from fancyimpute import KNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from scipy.stats import ks_2samp
from scipy import stats
from matplotlib import pyplot
from sklearn import preprocessing
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
import sqlite3
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as sklearnPCA
import plotly.plotly as py
from ggplot import *
from bokeh.charts import Scatter, Histogram, output_file, show
# from .theme import theme
# from .theme import theme_matplotlib
# from .theme_matplotlib import theme_matplotlib
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)



df = pd.read_csv("/Users/lmc2/practice_label.csv")
df2 = pd.read_csv("/Users/lmc2/practice_label.csv")
df3 = pd.read_csv("/Users/lmc2/tryX.csv")

###########################################
# Convert Categorical to Numerical
########################################### 
#Label Encoding: convert categorical to numerical
label_encoder = preprocessing.LabelEncoder()
df['all_label'] = label_encoder.fit_transform(df['all_label'])

print df.head()

###########################################
# Scale Data
########################################### 
df_norm = (df - df.mean()) / (df.max() - df.min())
NaN_count_post = df.isnull().sum()
print NaN_count_post


scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(df)


###########################################
# TSNE
########################################### 
tsne = TSNE(n_components=2, random_state=0)
Z = tsne.fit_transform(X)
dftsne = pd.DataFrame(Z, columns=['x','y'], index=df.index)
print dftsne.shape

###########################################
# Plots
########################################### 

'''
Add data to dataframe
'''

dftsne['all_label'] = df2['all_label']
dftsne['one_label'] = df2['one']
dftsne['two_label'] = df2['two']
dftsne['predicted_labels'] = df3['predicted_labels']
dftsne['Label'] = df3['Label']


'''
Generate plots
'''
p = Scatter(dftsne, x='x', y='y', title='HG002 INS: tSNE')
output_file("tSNE1_INS.html")
show(p)

p = Scatter(dftsne, x='x', y='y', color='all_label', title='HG002 CrowdVar: Percent Certainty (All)', legend="bottom_left")
output_file("tSNE_Percent_Certainty[All].html")
show(p)

p = Scatter(dftsne, x='x', y='y', color='one_label', title='HG002 CrowdVar: Percent Certainty (Group One)', legend="bottom_left")
output_file("tSNE_Percent_Certainty[one].html")
show(p)

p = Scatter(dftsne, x='x', y='y', color='two_label', title='HG002 CrowdVar: Percent Certainty (Group Two)', legend="bottom_left")
output_file("tSNE_Percent_Certainty[two].html")
show(p)

p = Scatter(dftsne, x='x', y='y', color='predicted_labels', title='HG002 CrowdVar: Model Predicted Labels', legend="bottom_left")
output_file("tSNE_predicted_labels.html")
show(p)

p = Scatter(dftsne, x='x', y='y', color='Machine_Label', title='HG002 CrowdVar: Crowd Sourced Labels', legend="bottom_left")
output_file("tSNE_CrowdSourced_labels.html")
show(p)

p = Histogram(log_size, values='INS_log_size', title='HG002 INS: Size Distribution [5000 Samples]', color='LightSlateGray', bins=19, xlabel="Size[log10]", ylabel="Frequency")
output_file("tSNE4_INS_Histo_logsize.html")
show(p)

p = Histogram(log_size, values='INS_log_size', title='HG002 INS: Size Distribution [5000 Samples]', color='LightSlateGray', bins=30, xlabel="Size[log10]", ylabel="Frequency")
output_file("tSNE4_INS_Histo_logsize.2.html")
show(p)

p = Scatter(dftsne, x='x', y='y', color='GTcons', title='HG002 INS: Consensus Genotypes', legend="top_left")
output_file("tSNE6_INS_GTcons.html")
show(p)

p = Scatter(dftsne, x='x', y='y', color='refN_pct', title='HG002 INS: Reference N', legend="top_left")
output_file("tSNE7_INS_RefN.html")
show(p)

p = Scatter(dftsne, x='x', y='y', color='size_bin', title='HG002 INS:Size', legend="top_left")
output_file("tSNE7_INS_SizeBin.html")
show(p)
