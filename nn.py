#import libraries
import tensorflow as tf
import numpy      as np
import pandas     as pd

#read dataframe
df = pd.read_csv('Dataset.csv')

#manipulate data to clean the dataset
df = df.drop(df[df['MARRIAGE'] < 1].index)

sex = pd.get_dummies(df['SEX'], drop_first=True)

df.drop('SEX',axis=1,inplace=True)
df.drop('ID',axis=1, inplace=True)

df = pd.concat([df,sex],axis=1)
df.loc[df.EDUCATION == 0,'EDUCATION'] = 3
df = df.rename(index=str, columns={2: 'SEX','PAY_0': 'PAY_1'})

#If we decide to consider unknown as other cathegory

#fil = (df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0)
#df.loc[fil, 'EDUCATION'] = 4
#df.EDUCATION.value_counts()

#split it Y: predicted X: with what we will predict
Y = df['default.payment.next.month']
X = df.drop('default.payment.next.month',axis=1)

# Parameters
learning_rate = 0.001
training_epochs = 15 #how many times we will iterate throught the dataset
batch_size = 100 #how many maincra 

# Network Parameters
n_hidden_1 = 32 
n_hidden_2 = 32 
n_input = len(df.columns)-1 
n_classes = 2  
n_samples = df.size/len(df.columns)


#70% for training 30% checking
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

#sd
import tensorflow.contrib.learn.python.learn as learn
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
classifier = learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[32, 32,32], n_classes=2)
classifier.fit(X_train, Y_train, steps=400, batch_size=10)

pred = classifier.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test,list(pred)))






