# pip3 install python-mnist
# mkdir datasets
# mkdir datasets/letters
# mkdir datasets/digits
# wget -O datasets/letters/train-images-idx3-ubyte.gz https://github.com/aurelienduarte/emnist/raw/master/gzip/emnist-letters-train-images-idx3-ubyte.gz
# wget -O datasets/letters/train-labels-idx1-ubyte.gz https://github.com/aurelienduarte/emnist/raw/master/gzip/emnist-letters-train-labels-idx1-ubyte.gz
# wget -O datasets/digits/train-images-idx3-ubyte.gz https://github.com/aurelienduarte/emnist/raw/master/gzip/emnist-digits-train-images-idx3-ubyte.gz
# wget -O datasets/digits/train-labels-idx1-ubyte.gz https://github.com/aurelienduarte/emnist/raw/master/gzip/emnist-digits-train-labels-idx1-ubyte.gz
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn import svm
from mnist import MNIST
import gzip
import shutil

with gzip.open('datasets/letters/train-images-idx3-ubyte.gz', 'rb') as f_in:
    with open('datasets/letters/train-images-idx3-ubyte', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
with gzip.open('datasets/letters/train-labels-idx1-ubyte.gz', 'rb') as f_in:
    with open('datasets/letters/train-labels-idx1-ubyte', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

with gzip.open('datasets/digits/train-images-idx3-ubyte.gz', 'rb') as f_in:
    with open('datasets/digits/train-images-idx3-ubyte', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
with gzip.open('datasets/digits/train-labels-idx1-ubyte.gz', 'rb') as f_in:
    with open('datasets/digits/train-labels-idx1-ubyte', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print("Datasets loaded!\n")

mndata = MNIST('datasets/letters')
x,y = mndata.load_training()
x = np.array(x)
y = np.array(y)
x = preprocessing.scale(x)
x_train = x[:10000]
x_test = x[60000:]
y_train = y[:10000]
y_test = y[60000:]
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

model = svm.SVC(gamma=0.001 , C = 100.)
print("Fitting model for Alphabets...")
model.fit(x_train,y_train)

filename = 'letter_model.sav'
pickle.dump(model, open(filename, 'wb'))
print("Done!\n")

mndata = MNIST('datasets/digits')
x,y = mndata.load_training()
x = np.array(x)
y = np.array(y)
x = preprocessing.scale(x)
x_train = x[:10000]
x_test = x[60000:]
y_train = y[:10000]
y_test = y[60000:]
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

model = svm.SVC(gamma=0.001 , C = 100.)
print("Fitting model for Digits...")
model.fit(x_train,y_train)

filename = 'digit_model.sav'
pickle.dump(model, open(filename, 'wb'))
print("Done!\n")