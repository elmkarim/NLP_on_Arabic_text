
# coding: utf-8

# ## Predicting text by learning from Quran corpus

# In[1]:


import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, SimpleRNN, GRU
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# In[2]:


# load ascii text
filename = "quran-simple-clean.txt"
raw_text = open(filename, encoding='utf-8').read()

#show 100 characters from text
raw_text[1:100]

#split to words
words=raw_text.split()
words.remove(words[0])


# In[3]:


## separate و of the words (add exception for words biginning by و)
Listw = [i for i in words if i.startswith("و") ]
Listw = list(set(Listw))

#Exceptions
exceptions = ['وكيل', 'وضع', 'وقفوا', 'واحدا', 'وقع', 'وسق', 'وقب', 'واسعة', 'وجد', 'وطرا', 'ودعك', 'وهاجا', 'واديا', 'وردا', 'وزر']
for ex in exceptions:
    Listw.remove(ex)

replacements = [(' '+w, ' '+w[0]+' '+w[1:]) for w in Listw]
for r in replacements:
    raw_text = raw_text.replace(r[0],r[1])


# In[4]:


# fit a tokenizer
from keras.preprocessing.text import Tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer(split='\n')
    tokenizer.fit_on_texts(lines)
    return tokenizer

# prepare tokenizer for commentaries
tokenizer = create_tokenizer(words)

#tokenizing words
# for i,token in enumerate(tokenizer.word_index, start=0):
#     print("{} : {}". format(i,token))
#     if i==10: break


# In[5]:


#Calculate vocabulary size
n_vocab = len(tokenizer.word_index) + 1
print('Vocabulary size (vocab_size):', n_vocab)


# In[6]:


#direct conversion function: from word to index
def word2index(word):
    return(tokenizer.word_index[word])
#reverse conversion function: from index to word
index2word = dict((i+1, word) for i, word in enumerate(tokenizer.word_index))


# In[21]:


# encode sequences
def encode_sequences(tokenizer, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    return X
X = encode_sequences(tokenizer, words)
print('Number of words in corpus: ', len(X))

# print(X[0:10])
# print([index2word[i[0]] for i in X[0:10]])


# In[8]:


# prepare the dataset of input to output pairs encoded as integers
# for each group of 100 words, predict the 101st following word in the text
n_words = len(words)
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_words - seq_length, 1):
    seq_in = words[i:i + seq_length]
    seq_out = words[i + seq_length]
    dataX.append([word2index(char) for char in seq_in])
    dataY.append(word2index(seq_out))
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)


# In[9]:


# i=1000
# print(dataX[i])
# print([index2word[w] for w in dataX[i]])
# print()
# print(dataY[i])
# print(index2word[dataY[i]] )


# In[10]:


# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


# In[20]:


# from sklearn.model_selection import train_test_split
# trainX, testX, trainY, testY = train_test_split(X, y,  test_size=0.1)
# print(trainX.shape[0],testX.shape[0],trainY.shape[0],testY.shape[0])


# In[19]:


import keras
import os

class modelRNN:
    def __init__(self, model_type):
        self.model_type = model_type

    def create_model(self, hidden_layer):
        model = Sequential()
        model.add(self.model_type(hidden_layer, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.summary()
        self.model = model
        return(self.model)
    
    def train_model(self, X, y, epochs, batch_size, weights_file, log_file, load_weights=True):
        if (load_weights) & (os.path.isfile(weights_file)):
            print('Loading weights from: ', weights_file)
            self.model.load_weights(weights_file)
        
        self.checkpoint = ModelCheckpoint(weights_file, monitor='loss', verbose=1, save_best_only=True, mode='min')
        self.csvlog = keras.callbacks.CSVLogger(log_file, separator=',', append=True)
        self.callbacks_list = [self.checkpoint, self.csvlog]
        print('Training model for %d epochs, batch_size=%d' % (epochs, batch_size))
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=self.callbacks_list, 
                  verbose=1, validation_split=0.1)


# In[ ]:


modelLSTM = modelRNN(model_type=keras.layers.GRU)
modelLSTM.create_model(hidden_layer=1000)
modelLSTM.train_model(X, y, epochs=100, batch_size=128, weights_file="weights_gru.hdf5",
                    log_file='results_gru.csv', load_weights=False )


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
# Get training and test loss histories
training_loss = training_loss + history.history['loss']
# test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'b')
# plt.plot(epoch_count, test_loss, 'b')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();



