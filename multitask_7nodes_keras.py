import numpy as np
import keras.backend as K
from keras.layers import Dense, Input
from keras.models import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from sklearn import datasets, preprocessing
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import argmax
from sklearn.metrics import classification_report, confusion_matrix
from keras.regularizers import l1,l2
# We have imported all dependencied
##df = pd.read_csv('dataset.csv') # read data set using pandas
import time #s tm
import os
from sklearn.metrics import accuracy_score
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_batch(dataset, i, BATCH_SIZE):
    if i*BATCH_SIZE+BATCH_SIZE > dataset.shape[0]:
            return dataset[i*BATCH_SIZE:, :]
    return dataset[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE), :]
    
def denormalize(df,norm_data):
    df = df['Close'].values.reshape(-1,1)
    norm_data = norm_data.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)



data = pd.read_csv('7nodesReg/result_n=7_250000sample.csv', header=-1, delimiter='\t')
#labels = pd.read_csv('encoded_flabels.csv', header=-1)
print (np.array(data).shape)

X = np.array(data)[:,0:-8]

Y1 = np.atleast_2d(np.array(data)[:,-7]).T
#Y1 = to_categorical(Y1)
Y2 = np.array(data)[:,-7:]
Y = np.concatenate((Y1,Y2),axis=1)
print (Y.shape)

scaler = MinMaxScaler() # For normalizing dataset
# We want to predict Close value of stock

x_trainer, x_tester, y_trainer, y_tester = train_test_split(X, Y, test_size = 0.2)

print (x_tester.shape)
print (y_tester.shape)
print (y_tester[0:2,:])


X_train = scaler.fit_transform(x_trainer)
y_train = y_trainer[:,1:]#scaler.fit_transform(y_trainer)
y_train_class = np.atleast_2d(y_trainer[:,0]).T
# y is output and x is features.
X_test = scaler.fit_transform(x_tester)
y_test = y_tester[:,1:]#scaler.fit_transform(y_test)
y_test_class = np.atleast_2d(y_tester[:,0]).T#np.atleast_2d(y_test[:,0]).T

##print (X_test.shape)
##print (y_test[0:5,:])
##print(y_test_class[0:5,:])


# Some random training data and labels
features = X_train
labels = y_train

# Simple neural net with three outputs
model = Sequential()
model.add(Dense(60, input_dim=42, activation='relu', kernel_initializer='he_uniform'))
#model.add(BatchNormalization())
model.add(Dense(20, activation='relu'))
model.add(Dense(40, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(7, activation='sigmoid'))

##model.add(Dense(15, input_dim=30, kernel_initializer='he_uniform'))
##model.add(BatchNormalization())
##model.add(Activation('relu'))
##model.add(Dense(10))
##model.add(BatchNormalization())
##model.add(Activation('relu'))
##model.add(Dense(5, activation='sigmoid'))

##input_layer = Input((30,))
##hidden_layer1 = Dense(15, activation='relu')(input_layer)
##model.add(BatchNormalization())
##
##hidden_layer2 = Dense(10, activation='relu')(hidden_layer1)
##model.add(BatchNormalization())
###hidden_layer3 = Dense(10, activation='relu')(hidden_layer2)
##output_layer = Dense(5,activation='sigmoid')(hidden_layer2)
### Model
##model = Model(inputs=input_layer, outputs=output_layer)

# Write a custom loss function
def custom_loss(y_true, y_pred):
    # Normal MSE loss
    mse = K.mean(K.square(y_true-y_pred), axis=-1)
    # Loss that penalizes differences between sum(predictions) and sum(labels)
    #sum_constraint = K.square(K.sum(y_pred, axis=-1) - K.sum(y_true, axis=-1))
    return(mse)#+sum_constraint)

error=[]
timer = []
acc = []
testimer = []
for i in range(1):
    # Compile with custom loss
    model.compile(loss=custom_loss, optimizer='adam')

    beg_tr = time.time()
    model.fit(features, labels, epochs=50, verbose=1)
    end_tr = time.time()
    print("elapsed training time: %f seconds" % (end_tr - beg_tr))
    timer.append(end_tr - beg_tr)
    
    #Testing
    beg_ts = time.time()
    prediction = model.predict(X_test)
    end_ts = time.time()
    print("elapsed testing time: %f seconds" % (end_ts - beg_ts))
    testimer.append(end_ts - beg_ts)
    
    mse = np.mean(np.square(prediction - y_test))#K.mean(K.square(y_true-y_pred), axis=-1)
    print("regression mean square error :", mse)
    error.append(mse)

    classifier = np.where(prediction < 0.15, 0, 1)
    X_test_class = np.where(y_test < 0.15, 0, 1)
    y_true = X_test_class.dot(1 << np.arange(X_test_class.shape[-1] - 1, -1, -1))
    y_pred = classifier.dot(1 << np.arange(classifier.shape[-1] - 1, -1, -1))
    print(classifier[0:10,:].dot(1 << np.arange(classifier[0:5,:].shape[-1] - 1, -1, -1)))
    print(X_test_class[0:10,:].dot(1 << np.arange(X_test_class[0:5,:].shape[-1] - 1, -1, -1)))
    print (accuracy_score(y_true, y_pred))
    acc.append(accuracy_score(y_true, y_pred))
    
class_pred = pd.DataFrame(y_pred)
class_label = pd.DataFrame(y_true)

reg_pred = pd.DataFrame(np.array(prediction))
reg_label= pd.DataFrame(y_test)

reslab = np.atleast_2d((['Acc', 'error', 'train time', 'Inference time']))

acc = np.atleast_2d((acc)).T
error = np.atleast_2d(error).T
timer = np.atleast_2d(timer).T
testimer = np.atleast_2d(testimer).T

vals = np.concatenate((acc,error,timer,testimer),axis=1)
result = np.concatenate((reslab,vals))
result = pd.DataFrame(result)

##class_pred.to_csv("7nodesReg/class_prediction_7nodes_50epochs_reg_th01.csv")
##class_label.to_csv("7nodesReg/class_label_7nodes_50epochs_reg_th01.csv")
##reg_pred.to_csv("7nodesReg/reg_prediction_7nodes_50epochs_reg_th01.csv")
##reg_label.to_csv("7nodesReg/reg_label_7nodes_50epochs_reg_th01.csv")
##result.to_csv("7nodesReg/results_50epochs_7nodes_reg_th01.csv")

print ("mse: ", error)
print ("Acc: ", acc)
print ("timer: ", timer)
print ("testimer: ", testimer)

# serialize model to JSON
model_json = model.to_json()
with open("7nodesReg/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
### later...
## 
### load json and create model
##json_file = open('model.json', 'r')
##loaded_model_json = json_file.read()
##json_file.close()
##loaded_model = model_from_json(loaded_model_json)
### load weights into new model
##loaded_model.load_weights("model.h5")
##print("Loaded model from disk")
## 
### evaluate loaded model on test data
##loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
##score = loaded_model.evaluate(X, Y, verbose=0)
##print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
