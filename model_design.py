import keras
import os
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, SimpleRNN, GRU
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

class modelRNN:
    def __init__(self, model_type):
        self.model_type = model_type

    def create_model(self, hidden_layer, input_shape, output_shape):
        self.model = Sequential()
        self.model.add(self.model_type(hidden_layer, input_shape=input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(output_shape, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model.summary()
        return(self.model)
    
    def train_model(self, X, y, epochs, batch_size, weights_file, log_file, load_weights=False):
        if (load_weights) & (os.path.isfile(weights_file)):
            print('Loading weights from: ', weights_file)
            self.model.load_weights(weights_file)
            self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        
        self.checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.csvlog = keras.callbacks.CSVLogger(log_file, separator=',', append=True)
        self.callbacks_list = [self.checkpoint, self.csvlog]
        print('Training model for %d epochs, batch_size=%d' % (epochs, batch_size))
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=self.callbacks_list, 
                  verbose=1, validation_split=0.1)
 
    def load_weights(self, weights_file):
        if os.path.isfile(weights_file):
            print('Loading weights from: ', weights_file)
            self.model.load_weights(weights_file)
        else:
            print('Weights file: ', weights_file, 'does not exist, skipping loading weights')
       
    def prediction(self, x, verbose=0):
        self.model.summary()
        pr = self.model.predict(x, verbose)
        return(pr)
    
