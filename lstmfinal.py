


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, GRU
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras import backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.initializers import Constant
from gensim.models import Word2Vec
import os
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow
# %matplotlib inline
pd.set_option('display.max_colwidth', 1000000)

with open('/content/drive/My Drive/Colab Notebooks/datf.pkl', 'rb') as f:
     dt  = pickle.load(f)

#Extract the same number of negative reviews as positive ones
df = dt.loc[dt['Class'] == 0.0]
df1 = df[0:80422]
df2 = dt.loc[dt['Class'] == 1.0]
df3 = pd.concat([df1, df2])
df3 = df3.sample(frac=1).reset_index(drop=True)
x = df3['Tweet'].values.tolist()
y = df3.Class



#Get the max length of a sentence
maxlen = max([len(s.split()) for s in x])
# Train word2vec model 
model = Word2Vec(x, size=100, min_count=1)
#Save the model
filename = 'word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)

# Get the embedding matrix

embedding_index = {}
f = open(os.path.join('','word2vec.txt'), encoding = "UTF-8")
for line in f:
  values=line.split()
  word=values[0]
  coefs=np.asarray(values[1:])
  embedding_index[word] = coefs
f.close

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
words_index = tokenizer.word_index
num_words = len(words_index) + 1
em = np.zeros((num_words, 100))
for word,i in words_index.items():
  if i > num_words:
    continue
  embedding_vector = embedding_index.get(word)
  if embedding_vector is not None:
    em[i] = embedding_vector

#Create the LSTM model
def RNN():
    inputs = Input(name='inputs',shape=maxlen)  
    layer = Embedding(num_words,100, embeddings_initializer=Constant(em), input_length=maxlen, trainable=False)(inputs)               
    layer = LSTM(100)(layer)
    layer = BatchNormalization()(layer)
    layer = Dense(35,name='FC1',kernel_initializer=GlorotNormal())(layer)
    layer = Activation('relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(35,name='FC2',kernel_initializer=GlorotNormal())(layer)
    layer = Activation('relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dense(35,name='FC3',kernel_initializer=GlorotNormal())(layer)
    layer = Activation('relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dense(35,name='FC4',kernel_initializer=GlorotNormal())(layer)
    layer = Activation('relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dense(35,name='FC5',kernel_initializer=GlorotNormal())(layer)
    layer = Activation('relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dense(35,name='FC6',kernel_initializer=GlorotNormal())(layer)
    layer = Activation('relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dense(2,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

# Create evaluation metrics
def recall_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = RNN()
model.summary()
model.compile(loss='CategoricalCrossentropy',optimizer=Nadam(),metrics=['acc',f1_score,precision_score, recall_score])

model.fit(x_train,y_train,batch_size=512,epochs=300,
          validation_split=0.2)

loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
print('Loss: %.3f, Accuracy: %.3f, F1_score: %.3f, Precision: %.3f, Recall: %.3f' % (loss, accuracy, f1_score, precision, recall))

# 2. Save Keras Model or weights on google drive

# create on Colab directory
model.save('model.h5')    
model_file = drive.CreateFile({'title' : 'model.h5'})
model_file.SetContentFile('model.h5')
model_file.Upload()

# download to google drive
drive.CreateFile({'id': model_file.get('id')})

#Create a file for weights
model.save_weights('model_weights.h5')
weights_file = drive.CreateFile({'title' : 'model_weights.h5'})
weights_file.SetContentFile('model_weights.h5')
weights_file.Upload()
drive.CreateFile({'id': weights_file.get('id')})

"""# New section"""
