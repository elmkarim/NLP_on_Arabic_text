import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer(split='\n')
    tokenizer.fit_on_texts(lines)
    return tokenizer

# #direct conversion function: from word to index
# def word2index(word):
#     return(tokenizer.word_index[word])


#Load serialized objects from pickle file
def load_from_pickle(filename):
    import pickle
    with open(filename, "rb") as f:
        return (pickle.load(f))

#Save serialized objects from pickle file
def save_to_pickle(filename, objects):
    import pickle
    with open(filename, "wb") as f:
        pickle.dump(objects, f)

# encode sequences
def encode_sequences(tokenizer, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    return X

def create_data(filename):
    # load ascii text
    raw_text = open(filename, encoding='utf-8').read()

    #split to words
    words=raw_text.split()
    words.remove(words[0])

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

    # prepare tokenizer for commentaries
    tokenizer = create_tokenizer(words)

    #Calculate vocabulary size
    n_vocab = len(tokenizer.word_index) + 1
    print('Vocabulary size (vocab_size):', n_vocab)

    #reverse conversion function: from index to word
    index2word = dict((i+1, word) for i, word in enumerate(tokenizer.word_index))
    word2index = dict((word, i+1) for i, word in enumerate(tokenizer.word_index))

    X = encode_sequences(tokenizer, words)
    print('Number of words in corpus: ', len(X))

    # prepare the dataset of input to output pairs encoded as integers
    # for each group of 100 words, predict the 101st following word in the text
    n_words = len(words)
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_words - seq_length, 1):
        seq_in = words[i:i + seq_length]
        seq_out = words[i + seq_length]
        dataX.append([word2index[char] for char in seq_in])
        dataY.append(word2index[seq_out])
    n_patterns = len(dataX)
    print ("Total Patterns: ", n_patterns)

    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

    # normalize
    X = X / float(n_vocab)

    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)
    
    return(words, X, dataX, dataY, n_words, n_vocab, index2word, word2index)

def create_save_data(filename, picklefile):
    words, X, dataX, dataY, n_words, n_vocab, index2word, word2index = create_data(filename)
    save_to_pickle(picklefile, (words, X, dataX, dataY, n_words, n_vocab, index2word, word2index))
    print('data saved to pickle file:', picklefile)
    
def read_data_from_pickle(picklefile):
    words, X, dataX, dataY, n_words, n_vocab, index2word, word2index = load_from_pickle(picklefile)
    y = np_utils.to_categorical(dataY)
    return(words, X, dataX, y, n_words, n_vocab, index2word, word2index)


def read_train_set_from_pickle(picklefile):
    X_train, X_test, y_train, y_test = load_from_pickle(picklefile)
    return (X_train, X_test, y_train, y_test)    
    
def create_save_train_set(original_picklefile, trainset_picklefile, test_size=0.2):
    #Create training and test sets
    words, X, dataX, y, n_words, n_vocab, index2word, word2index = read_data_from_pickle(original_picklefile)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    save_to_pickle(trainset_picklefile, (X_train, X_test, y_train, y_test))
    print ('Oringinal set:', X.shape, y.shape)
    print ('Training set:', X_train.shape, y_train.shape)
    print ('Test set:', X_test.shape, y_test.shape)
    print('Training and test sets saved to:', trainset_picklefile)
                

