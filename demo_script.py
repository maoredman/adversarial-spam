import os
import glob
import pickle
import sys
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def read_data(path_name):
    error_counts = 0
    data_list = []
    fn_list = []
    for fn in os.listdir(path_name):
        print(fn)
        try:
            with open(os.path.join(path_name, fn),'r', encoding='utf8', errors='ignore') as f:
                data = f.read()
            data_list.append(data)
            fn_list.append(fn)
        except Exception as e:
            print(fn, e)
            error_counts += 1
    print('Error Counts: ', error_counts)
    print(len(fn_list), 'mail read')
    return (data_list, fn_list)

if __name__ == '__main__':

    SEQ_LEN = 1000
    model_path = './models/spam_model.h5'
    tokenizer_path = './models/tokenizer.pkl'
    
    ## Loading Data    
    print('Loading Data...')
    data, files = read_data(sys.argv[1])

    ## Loading Model and Tokenizer
    print('Loading Model and Tokenizer...')
    with open(tokenizer_path,'rb') as f:
        tokenizer = pickle.load(f)
    model = load_model(model_path)
    
    ## Preprocessing
    dl_x = tokenizer.texts_to_sequences(data)
    dl_x = pad_sequences(dl_x, maxlen = SEQ_LEN)
    
    ## Model Predict
    print('Model Predict...')
    pred = model.predict(dl_x)
    for i, yp in enumerate(pred):
        print(files[i], yp[1], 0 if yp[1]<0.5 else 1)
