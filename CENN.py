#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jose Manuel Casas: casasjm@uniovi.es
CENN paper: https://arxiv.org/abs/2205.05623: 
"""

# CENN modified by Laura & Sara

# This module defines the Cosmic microwave background
# extraction neural network (CENN) architecture and trains it
# For using it: python CENN_Execute.py <Number of GPU>

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU, Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, MaxPool2D, BatchNormalization, Activation, Input
import os
import argparse
import h5py
import numpy as np
import json
import matplotlib.pyplot as plt

print (tf.__version__)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('ERROR')


##############################################################################

# Original hyperparameters set

# learning_rate = 0.05
# batch_size = 32
# num_epochs = 500
# regularizer = keras.regularizers.l2(0.00001)
# activation_function = tf.keras.layers.LeakyReLU(alpha=0.2)
# optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
# loss = keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)

##############################################################################

# def parse_arguments():
    
#     parser = argparse.ArgumentParser()

#     parser.add_argument('GPU', type=str, help = 'Number of GPU you want to use')
#     args = parser.parse_args()
    
#     return args

def error(y_pred, y_true):
    
    # We use the MSE error along the paper
    return K.mean(K.square(y_pred - y_true), axis=-1)

def build_model_Conv(nm_NNpar, nm_par):
    with open(nm_par) as par_file:
        Par = json.loads(par_file.read())
    input_image_size = Par["Patch_Size"]

    with open(nm_NNpar) as par_file:
        NNpar = json.loads(par_file.read())
    
    learning_rate = NNpar["learning_rate"]
 
    L2_regularizer_value = NNpar["L2_regularizer_value"]
    regularizer = keras.regularizers.l2(L2_regularizer_value)
    # regularizer = keras.regularizers.l2(0.000001)
    activation_function = tf.keras.layers.LeakyReLU(alpha=0.2)
    #activation_function = tf.keras.layers.ReLU(max_value = None, negative_slope = 0.0,threshold=0.0)
    channels_order = NNpar["channels_order"]
    padding_type =  NNpar["padding_type"]
  
    # Patches of 256x256 pixels and 3 frequency input channels
  
    inputs = Input(shape=(input_image_size, input_image_size, 3)) 
  
    conv1 = Conv2D(filters = 8, kernel_size = 9, strides = 2, padding=padding_type, activation=None,
                   data_format=channels_order, kernel_regularizer = regularizer)(inputs)
  
    conv1_activation_function = activation_function(conv1)

    conv2 = Conv2D(filters = 16, kernel_size = 9, strides = 2, padding=padding_type, activation=None,
                   data_format=channels_order, kernel_regularizer = regularizer)(conv1_activation_function)
  
    conv2_activation_function = activation_function(conv2)
  
    conv3 = Conv2D(filters = 64, kernel_size = 7, strides = 2, padding=padding_type, activation=None,
                   data_format=channels_order, kernel_regularizer = regularizer)(conv2_activation_function)
  
    conv3_activation_function = activation_function(conv3)
  
    conv4 = Conv2D(filters = 128, kernel_size = 7, strides = 2, padding=padding_type, activation=None,
                   data_format=channels_order, kernel_regularizer = regularizer)(conv3_activation_function)
  
    conv4_activation_function = activation_function(conv4)
  
    conv5 = Conv2D(input_image_size, 5, 2, 'same', activation=None, data_format=channels_order,kernel_regularizer = regularizer)(conv4_activation_function)
  
    conv5_activation_function = activation_function(conv5)
  
    conv6 = Conv2D(512, 3, 2, 'same', activation=None, data_format=channels_order,kernel_regularizer = regularizer)(conv5_activation_function)

    conv6_activation_function = activation_function(conv6)

    deconv1 = Conv2DTranspose(input_image_size, 3, 2, 'same', data_format=channels_order, activation=None, kernel_regularizer = regularizer)(conv6_activation_function)
  
    deconv1_activation_function = activation_function(deconv1)
  
    add1 = tf.keras.layers.Concatenate(axis=3)([conv5, deconv1_activation_function])
  
    deconv2 = Conv2DTranspose(128, 5, 2, 'same', data_format=channels_order, activation=None, kernel_regularizer = regularizer)(add1)
  
    deconv2_activation_function = activation_function(deconv2)
  
    add2 = tf.keras.layers.Concatenate(axis=3)([conv4, deconv2_activation_function])
  
    deconv3 = Conv2DTranspose(filters = 64, kernel_size = 7, strides = 2, padding=padding_type,
                              data_format=channels_order, activation=None,
                              kernel_regularizer = regularizer)(conv4_activation_function)#(add2)
  
    deconv3_activation_function = activation_function(deconv3)
  
    add3 = tf.keras.layers.Concatenate(axis=3)([conv3, deconv3_activation_function])
  
    deconv4 = Conv2DTranspose(filters = 16, kernel_size = 7, strides = 2, padding=padding_type,
                              data_format=channels_order, activation=None,
                              kernel_regularizer = regularizer)(add3)
  
    deconv4_activation_function = activation_function(deconv4)
  
    add4 = tf.keras.layers.Concatenate(axis=3)([conv2, deconv4_activation_function])
  
    deconv5 = Conv2DTranspose(filters = 8, kernel_size = 9, strides = 2, padding=padding_type,
                              data_format=channels_order, activation=None,
                              kernel_regularizer = regularizer)(add4)
  
    deconv5_activation_function = activation_function(deconv5)
  
    add5 = tf.keras.layers.Concatenate(axis=3)([conv1, deconv5_activation_function])
  
    deconv6 = Conv2DTranspose(filters = 1, kernel_size = 9, strides = 2, padding=padding_type,
                              data_format=channels_order, activation=None,
                              kernel_regularizer = regularizer)(add5)
  
    deconv6_activation_function = activation_function(deconv6)
  
    model = tf.keras.Model(inputs, deconv6_activation_function)
  
    optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
    loss = keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
  
    model.compile(loss="mean_squared_error",
                  optimizer=optimizer,
                  metrics=[loss])

    return model

def CENN_train(nm_NNpar, nm_par):
    # CENN_train("./Parfile_CENN.par", "./Parfile_U.par")

    from datetime import datetime
    import json

    print(str(datetime.now()))

    with open(nm_NNpar) as par_file:
        NNpar = json.loads(par_file.read())
    batch_size = NNpar["batch_size"]
    num_epochs = NNpar["num_epochs"]
    test_frequency = NNpar["test_frequency"]
    learning_rate = NNpar["learning_rate"]
    
    with open(nm_par) as par_file:
        Par = json.loads(par_file.read())
    Patch_Size = Par["Patch_Size"]
    Filtro = Par["Filtro"]
    train_file = Par["train_file"]
    test_file =  Par["test_file"]
 
    inputs_test, labels_test, inputs_train, labels_train = read_the_data(train_file, test_file, Patch_Size)
    
    model = build_model_Conv(nm_NNpar, nm_par)  

    model.summary()

    Checkpoint = keras.callbacks.ModelCheckpoint('Models_'+Filtro+'/'+train_file[26:-3]+'_checkpoint-{val_loss:.5f}-{epoch:02d}.h5', monitor='val_loss', save_best_only=True)
    Best = keras.callbacks.ModelCheckpoint('Models_'+ Filtro+'/'+train_file[26:-3]+'_Red_'+ Filtro +'.h5', monitor='val_loss', save_best_only=True)

    # history = model.fit(inputs_train, labels_train, batch_size = batch_size, shuffle=True,
    #           epochs = num_epochs, verbose = 1, validation_freq = test_frequency,
    #           validation_data = (inputs_test, labels_test), callbacks=[Checkpoint, Best])

    history = model.fit(inputs_train, labels_train, batch_size = batch_size, shuffle=True,
                        epochs = num_epochs, verbose = 1, validation_freq = test_frequency,
                        validation_data = (inputs_test, labels_test), callbacks=[Best])

    print(history.history.keys())
    #  "Accuracy"
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    # plt.savefig('./Models_'+Filtro+'/'+validation_file[26:-3]+'_Accuracy_'+Filtro+'.pdf')
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    plt.savefig('./Models_'+Filtro+'/'+train_file[26:-3]+'_Loss_'+Filtro+'.pdf')
    
    results = model.predict(inputs_test)
    #loss_error = error(results, labels_test)   #porqe esta comentado?
    print(str(datetime.now()))

    pass


def CENN_TestValidation(nm_par):
    # POSPEN_TestValidation("./Parfile_Q.par")
    
    from datetime import datetime
    import json

    print(str(datetime.now()))

    with open(nm_par) as par_file:
        Par = json.loads(par_file.read())
    Patch_Size = Par["Patch_Size"]
    Filtro = Par["Filtro"]
    train_file = Par["train_file"]
    validation_file = Par["validation_file"]

    ###### to be checked
    # best_model_nm = Par["best_model_nm"]
    out_file = './Models_'+Filtro+'/'+validation_file[26:-3]+'_Outputs_CENN_'+Filtro+'.h5'

    # normalised validation dataset
    Model_file_path = './Models_'+Filtro+'/'+train_file[26:-3]+'_Red_'+Filtro+'.h5'
    print(Model_file_path)

    validation = h5py.File(validation_file, 'r')

    inputs, labels, maximo, minimo = normalize(validation, Patch_Size, maxmin = True)  

    model = keras.models.load_model(Model_file_path)
    model.summary()

    results = model.predict(inputs)
    
    results_denorm = denorm(results, maximo, minimo, Patch_Size)

    with h5py.File(out_file, 'w') as f:
        f['M'] = results_denorm
    
    print(str(datetime.now()))

    pass

    # save_data(results_denorm, labels_validation_no_normalized, name_output_file='Outputs_CENN_'+Filtro+'.h5')

#if __name__ == "__main__":
    #args = parse_arguments()
    #os.environ["CUDA_VISIBLE_DEVICES"]= args.GPU
    #train(batch_size, num_epochs, test_frequency, train_file_path, test_file_path, Patch_Size, Filtro)

def denorm(results, maximo, minimo, Patch_Size):
    
    denorm_output = np.zeros([len(results), Patch_Size, Patch_Size, 1])
    # denorm_output_dummy = np.zeros([len(results), Patch_Size, Patch_Size, 1])

    for i in range (0, len(results)):

        denorm_output[i,:,:,0] = (results[i,:,:,0] * (maximo[i] - minimo[i])) + minimo[i]

    return denorm_output
    
def read_the_data(train, test, Patch_Size):
    
    # Train
    
    train_file = h5py.File(train, 'r')

    inputs_train, labels_train = normalize(train_file, Patch_Size)
    
    test_file = h5py.File(test, 'r')

    inputs_test, labels_test = normalize(test_file, Patch_Size)
    
    return inputs_test, labels_test, inputs_train, labels_train


def normalize(inp_file, Patch_Size, maxmin = False):    
    inputs = inp_file["M"][:,:,:,:]
    labels = inp_file["M0"][:,:,:,:]
    
    normalize_inputs = np.zeros(
            [len(inputs), Patch_Size, Patch_Size, 3])
       
    normalize_labels = np.zeros(
        [len(labels), Patch_Size, Patch_Size, 1])

    if maxmin:
        maximo = np.zeros([len(inputs)])
        minimo = np.zeros([len(inputs)])
    
    for i in range(len(inputs)):
        
        min_labels = np.min(labels[i,:,:,0])
        max_labels = np.max(labels[i,:,:,0])
   
        min_inputs_low = np.min(inputs[i,:,:,0])
        max_inputs_low = np.max(inputs[i,:,:,0])
        min_inputs = np.min(inputs[i,:,:,1])
        max_inputs = np.max(inputs[i,:,:,1])
        min_inputs_high = np.min(inputs[i,:,:,2])
        max_inputs_high = np.max(inputs[i,:,:,2])

        normalize_inputs[i,:,:,0] = (inputs[i,:,:,0] - min_inputs_low)/(max_inputs_low - min_inputs_low)
        normalize_inputs[i,:,:,1] = (inputs[i,:,:,1] - min_inputs)/(max_inputs - min_inputs)
        normalize_inputs[i,:,:,2] = (inputs[i,:,:,2] - min_inputs_high)/(max_inputs_high - min_inputs_high)
        normalize_labels[i,:,:,0] = (labels[i,:,:,0] - min_inputs)/(max_inputs - min_inputs)

        if maxmin:
            maximo[i] = max_inputs
            minimo[i] = min_inputs
    if  maxmin:
        return normalize_inputs, normalize_labels, maximo, minimo
    else:
        return normalize_inputs, normalize_labels

def read_validation_data(h5_file_path, Patch_Size):
    
    h5_file = h5py.File(h5_file_path, 'r')

    inputs = h5_file["M"][:,:,:]
    
    labels = h5_file["M0"][:]
    labels = np.ndarray.flatten(labels)

    return labels, inputs
################################
################################
def plot_validaton_simu(nm_par,nsim):

    with open(nm_par) as par_file:
        Par = json.loads(par_file.read())
    Patch_Size = Par["Patch_Size"]
    Filtro = Par["Filtro"]
    train_file = Par["train_file"]
    validation_file = Par["validation_file"]

    ###### to be checked
    # best_model_nm = Par["best_model_nm"]
    out_file = './Models_'+Filtro+'/'+validation_file[26:-3]+'_Outputs_CENN_'+Filtro+'.h5'

    
    labels, inputs = read_validation_data(train_file, Patch_Size)

    h5_file = h5py.File(out_file, 'r')
    outpus = h5_file["M"][:,:,:]

    figure(1)

    fig, ([ax0, ax1, ax2]) = plt.subplots(1, 3, figsize=(16,6))
    fig.tight_layout(pad=4.0)
            
    ax0.title.set_text('Input Total')
    ax1.title.set_text('Label')
    ax2.title.set_text('Output')

    fig0=ax0.imshow((inputs[nsim,:,:,0]))
    fig.colorbar(fig0, ax=ax0, fraction=0.046, pad=0.04)
            
    fig1=ax1.imshow((labels[nsim,:,:,0]))
    fig.colorbar(fig1, ax=ax1, fraction=0.046, pad=0.04)
            
    fig2=ax2.imshow(outputs[nsim,:,:,0])
    fig.colorbar(fig2, ax=ax2, fraction=0.046, pad=0.04)

    savefig('./Models_'+Filtro+'/'+validation_file[26:-3]+'_Outputs_CENN_'+Filtro+'_'+str(nsim)+'.pdf')

pass

################################
################################

def main():
    #CENN_train("./Parfile_CENN.par", "./Parfile_I.par")
    CENN_TestValidation("./Parfile_I.par")


if __name__ == "__main__":
    main()
