import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from tensorflow.python.lib.io import file_io
from tensorflow import __version__ as tf_version

from datetime import datetime
import time
import pickle
import argparse

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM
from keras import optimizers
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


batch_size = 100
num_classes = 2
epochs = 2000

# Create a function to allow for different training dat and other options
def train_model(train_file='sentiment_set.pickle', 
                job_dir='./tmp/example-5', 
                dropout_one = 0.5,
                dropout_two = 0.5,
                **args):
    # set the logging path for ML Engine logging to Storage Bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('-----------------------')
    print('Using train_file located at {}'.format(train_file))
    print('Using logs_path located at {}'.format(logs_path))
    print('-----------------------')

    # set different mode of file_io.FileIO for different tf version
    if tf_version >= '1.1.0':
        mode = 'rb'
    else:
        mode = 'r'

    file_stream = file_io.FileIO(train_file, mode=mode)
    x_train, y_train, x_test, y_test  = pickle.load(file_stream)
    
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    
    x_train /= np.max(x_train)
    x_test /= np.max(x_test)

    print(x_train.shape, y_train.shape, 'train samples,', type(x_train[0][0]), ' ', type(y_train[0][0]))
    print(x_test.shape,  y_test.shape,  'test samples,',  type(x_test[0][0]),  ' ', type(y_train[0][0]))
    
    # to add LSTM layer need to reshape x_train, x_test to 
    # [n_samples, n_timestamps, n_outdims] where n_timestamps = 1
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))


    model = Sequential()
    model.add(LSTM(16, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_one))

    model.add(Dense(8))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_two))

    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    opt = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, 
                          verbose=1, epsilon=1e-4, mode='min')
    ]

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks
                        )
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    model.save('model.h5')
    
    # Save model.h5 on to google storage
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input arguments
    parser.add_argument(
      '--train-file',
      help='GCS or local paths to training data',
      required=True
    )
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    parser.add_argument(
      '--dropout_one',
      help='Dropout hyperparameter after the first dense layer'
    )
    parser.add_argument(
      '--dropout_two',
      hepp='Dropout hyperparam after the second dense layer'
    )

    args = parser.parse_args()
    arguments = args.__dict__
    
    train_model(**arguments)
