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
print(df)
df = pd.concat([df,sex],axis=1)
df.loc[df.EDUCATION == 0,'EDUCATION'] = 4 #others
df = df.rename(index=str, columns={2: 'SEX','PAY_0': 'PAY_1'})

#If we decide to consider unknown as other cathegory

#fil = (df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0)
#df.loc[fil, 'EDUCATION'] = 4
#df.EDUCATION.value_counts()

#split it Y: predicted X: with what we will predict
Y = df['default.payment.next.month']
X = df.drop('default.payment.next.month',axis=1)

#70% for training 30% checking
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30,random_state=101)

#sd
import tensorflow.contrib.learn.python.learn as learn

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
classifier = learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[32, 32], n_classes=2)
classifier.fit(X_train, Y_train, steps=30000, batch_size=1000)

pred = classifier.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test,list(pred)))

'''
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

#placeholders
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

#MultiLayer Model
def multilayer_perceptron(x, weights, biases):
    
    x : Place Holder for Data Input
    weights: Dictionary of weights
    biases: Dicitionary of biases
    
    
    # First Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    # Second Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    # Last Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

#Bias and Weights
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

#Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#initializing variables
init = tf.initialize_all_variables()

sess = tf.InteractiveSession()
sess.run(init)

for epoch  in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples/batch_size)
    
    for i in range(total_batch):
        batch_x, batch_y = df.train.next_batch(batch_size)
        #sess.run returns a tuple, but we need just the c, so we put the _ as a throwaway
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

        avg_cost += c/total_batch
    print("Epoch: {} cost={:.4f}".format(epoch+1,avg_cost))
print("Model has completed {} Epochs of Training".format(training_epochs))

'''