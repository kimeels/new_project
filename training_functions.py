import tensorflow as tf
import numpy as np 
import os
import pickle

import logging
tf.get_logger().setLevel(logging.ERROR)


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
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def make_model_fine_tune(final_layer = 6, learning_rate=0.01, fine_tune=False, weights_path=None):
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

    if fine_tune:
        model.load_weights(weights_path)   # for fine-tuning
        # This should freeze and unfreeze all of the layers apart from
        # BatchNormalization

        # Only allow training on the final layer
        for layer in model.layers[:-1]:
            layer.trainable = False
        

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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

    # checkpoint_path = dirname+"cp-{epoch:04d}.ckpt"
    checkpoint_path = os.path.join(dirname, 'cp.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    logger_path = os.path.join(dirname, 'training.log')

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=save_weights_only,
                                                     save_best_only=True,
                                                     verbose=verbose)

    # LC add a few more callbacks
    # Output training data after each epoch
    lg_callback = tf.keras.callbacks.CSVLogger(logger_path)
    # Reduce learning rate by factor every patience epoch without
    # change in val_loss
    rl_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.1,
                                                       patience=5,
                                                       mode='min',
                                                       min_delta=0.0,
                                                       min_lr=1e-5)
    # Stop training after patience epochs without improvement in val_loss
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=10,
                                                   mode='min',
                                                   min_delta=0.0)

    # If we use a generator then y_val is not specified
    if y_val is None:
        val = x_val
    else:
        val = (x_val, y_val)
        
    if gpu:
        with tf.device('/device:GPU:0'):
            history = model.fit(x_train, y_train,
                                validation_data=val,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=verbose,
                                callbacks=[cp_callback,
                                           lg_callback,
                                           rl_callback,
                                           es_callback])
                            
                                # max_queue_size=16,
                                # workers=8,
                                # use_multiprocessing=True)
    else:
        history = model.fit(x_train, y_train,
                            validation_data=val,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[cp_callback])

    pickle.dump(history.history['loss'], open( dirname+"loss.p", "wb" ) )
    pickle.dump(history.history['val_loss'], open( dirname+"val_loss.p", "wb" ) )


# def fine_tune_network(model,x_train,y_train,x_val,y_val,dirname='training_root', batch_size = 128,epochs = 200,
#                       save_weights_only = False,verbose = 1, gpu = True):
