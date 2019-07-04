
# coding: utf-8

# ## Predicting text by learning from Quran corpus

# In[1]:


#import python file that contains all functions for data creation and retrieval
import data_functions
import keras


# In[2]:


#Load data from already created pickle file
words, X, dataX, y, n_words, n_vocab, index2word, word2index = data_functions.read_data_from_pickle("data.pickle")


# In[6]:


models ={'rnn' : {'model': keras.layers.SimpleRNN, 'weights_file':'weights_rnn.hdf5', 'log_file': 'results_rnn.csv'},
         'lstm': {'model': keras.layers.LSTM, 'weights_file': 'weights_lstm.hdf5', 'log_file': 'results_lstm.csv'},
         'gru' : {'model': keras.layers.GRU, 'weights_file':'weights_gru.hdf5', 'log_file': 'results_gru.csv'}
        }


# In[9]:


from model_design import modelRNN
from importlib import reload 
# reload(model_design)

mdl = 'rnn'
model = modelRNN(model_type=models[mdl]['model'])
model_lstm=model.create_model(hidden_layer=2000, input_shape=(X.shape[1], X.shape[2]), output_shape=y.shape[1])
model.load_weights(weights_file=models[mdl]['weights_file'])
model.train_model(X, y, epochs=100, batch_size=128, weights_file=models[mdl]['weights_file'],
                    log_file=models[mdl]['log_file'], load_weights=False )


# In[ ]:


# from model_design import modelRNN
# import keras
# from importlib import reload 
# # reload(model_design)

# model = modelRNN(model_type=keras.layers.LSTM)
# model.create_model(hidden_layer=1000, input_shape=(X.shape[1], X.shape[2]), output_shape=y.shape[1])
# model.load_weights(weights_file="weights-quran-all-tmp.hdf5")
# model.train_model(X, y, epochs=100, batch_size=128, weights_file="weights-quran-all-tmp.hdf5",
#                     log_file='results.csv', load_weights=False )


# In[ ]:


# %matplotlib inline
# import matplotlib.pyplot as plt
# # Get training and test loss histories
# training_loss = training_loss + history.history['loss']
# # test_loss = history.history['val_loss']

# # Create count of the number of epochs
# epoch_count = range(1, len(training_loss) + 1)

# # Visualize loss history
# plt.plot(epoch_count, training_loss, 'b')
# # plt.plot(epoch_count, test_loss, 'b')
# plt.legend(['Training Loss', 'Test Loss'])
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show();


# In[ ]:


# !jupyter nbconvert --to script Udacity_capstone.ipynb

