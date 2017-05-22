

'''
Imports
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

'''
Uploads and modify data
'''
df = pd.read_csv('/Users/lmc2/Desktop/NIHFAES/FinalProject/Train/Data/CrowdVar.Train_250bp_HG002.csv') 
X = pd.read_csv('/Users/lmc2/Desktop/NIHFAES/FinalProject/Train/Data/CrowdVar.Train_250bp_HG002.csv')
X.drop(["sample", "chrom", "CN0_prob", "CN1_prob", "CN2_prob", "GTcons", "GTconflict", "GTsupp"], axis=1, inplace=True)
X = X.dropna()
X2 = X.dropna()
Y = X.pop('Label')
# X.drop(["sample"], axis=1, inplace=True)
X.head()

'''
Split dataset into training and testing set. 
'''

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=0)


'''
Train machine learning classifier - Randfom Forest Classifier
'''

model = RandomForestClassifier() 
#out of bag samples to estimate general accuracy
model.fit(X_train, y_train)

'''
Use the trained model to predict labels for the test set
'''
pred = model.predict_proba(X_test)
X6 = pd.concat([X_test, pd.DataFrame(pred, columns=['1','2','3'])])
X4 = pd.DataFrame(pred, columns=['1','2','3'])
X5 = pd.concat([X4,X_test, y_test], axis=1)
print X6.head()

X6.to_csv('try1.csv', index=False)

df_pred = pd.read_csv('/Users/lmc2/try3.csv')
X8 = pd.concat([df_pred, y_test], axis=1)

X9 = pd.concat([df_pred, pd.DataFrame(y_test, columns=['label'])])

X10 = pd.concat([X_test, pd.DataFrame(y_test, columns=['label'])])

labeled = pd.DataFrame()
labeled = y_test
labeled.to_csv('tryL.csv', index=False, header=True)

df_label = pd.read_csv('/Users/lmc2/tryL.csv')


X12 = pd.concat([df_pred, df_label], axis=1)
X12.to_csv('tryX12.csv', index=False, header=True)

predict_label = model.predict(X_test)
predict_label

X_test['predicted_labels'] = predict_label
X_test['true_labels'] = y_test

X12['predicted_labels'] = predict_label

X12.to_csv('tryX.csv', index=False, header=True)


'''
Use f1_score to determine the overall performance of the machine learning model. Essentially, generate a metric to show how well the model predicted each label
'''
y_true = X_test['Ill250.GT']
y_pred = X_test['predicted_labels']
f1_score(y_true, y_pred, average=None)