import tensorflow as tf
import numpy as np 
import pylab as pl 
import seaborn as sns
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def make_model(final_layer = 6):
    """
    Creates a network with the architecture specified in the paper. Final layer size is changable depending
    the number of parameters you want to infer.

    Parameters:
    -----------
    final_layer : int
        The number of neurons in the final layer. Depends on number of parameters to infer. Matches the shape training labels.

    """
    initializer = tf.keras.initializers.GlorotNormal()


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),padding ='same',kernel_initializer=initializer,use_bias =False))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),padding ='same',kernel_initializer=initializer,use_bias =False))
    model.add(tf.keras.layers.BatchNormalization(beta_initializer=initializer,momentum = 0.9))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),padding ='same',kernel_initializer=initializer,use_bias =False))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),padding ='same',kernel_initializer=initializer,use_bias =False))
    model.add(tf.keras.layers.BatchNormalization(beta_initializer=initializer,momentum = 0.9))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),padding ='same',kernel_initializer=initializer,use_bias =False))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),padding ='same',kernel_initializer=initializer,use_bias =False))
    model.add(tf.keras.layers.BatchNormalization(beta_initializer=initializer,momentum = 0.9))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1),padding ='same',kernel_initializer=initializer,use_bias =False))
    model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1),padding ='same',kernel_initializer=initializer,use_bias =False))
    model.add(tf.keras.layers.BatchNormalization(beta_initializer=initializer,momentum = 0.9))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1024,kernel_initializer=initializer,use_bias =False))
    model.add(tf.keras.layers.BatchNormalization(beta_initializer=initializer,momentum = 0.9))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Dense(1024,kernel_initializer=initializer,use_bias =False))
    model.add(tf.keras.layers.BatchNormalization(beta_initializer=initializer,momentum = 0.9))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Dense(1024,kernel_initializer=initializer,use_bias =False))
    model.add(tf.keras.layers.BatchNormalization(beta_initializer=initializer,momentum = 0.9))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Dense(final_layer,kernel_initializer=initializer,use_bias =False))

    model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def train_network(model,x_train,y_train,x_val,y_val,dirname='training_root', batch_size = 128,epochs = 200,
                  save_weights_only = False,verbose = 1, gpu = True):
    """
    Quick script to train the model

    Parameters:
    -----------
    model : tensorflow object
        CNN model, made using make_model() function.
    x_train : array
        Input training data.
    y_train : array
        Input training labels.
    x_val : array
        Input validation data.
    y_val : array
        Input validation labels.
    dirname : str
        Name of directory to save training steps.
    batch_size : int
        Size of training batch.
    epochs : int
        Number of epochs to run training for.
    save_weights_only : boolean
        If False, will save entire network so requires much more storage.
    verbose : int
        Output message detail.
    gpu : boolean
        If True, will attempt to use gpu to train.

    """

    dirname = "paper1_hassan/"
    checkpoint_path = dirname+"cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)


    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=save_weights_only,
                                                     verbose=verbose)

    if gpu:
        with tf.device('/device:GPU:0'):
            history = model.fit(x_train, y_train,
                                validation_data = (x_val,y_val),
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  verbose=verbose,
                                  callbacks=[cp_callback])
    else:
        history = model.fit(x_train, y_train,
                            validation_data = (x_val,y_val),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[cp_callback])

    pickle.dump(history.history['loss'], open( dirname+"loss.p", "wb" ) )
    pickle.dump(history.history['val_loss'], open( dirname+"val_loss.p", "wb" ) )
