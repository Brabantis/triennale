#!/usr/bin/env python
"""
Generate a neural network from a dataset
"""
# Make sure that theano.config.openmp is set to True for multiprocessor elaboration
import tables
import numpy as np
import argparse
import os
import h5py
import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Reshape, Merge
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization

from bcfind.util import bot_alert as ba

class CellularData(object):
    def __init__(self, filename, batchsize=128, start=0, end=None, p_shape=(32,32)):
        #With these settings it coughs up patches of 32X32, apparently. Is that right?
        h5file = h5py.File(filename, 'r')
        self.X = h5file['X'][start:end,:,:]
        self.y = h5file['y'][start:end,:,:]
        if end is None:
            end = self.X.shape[0]
        assert self.X.shape == self.y.shape
        self.batchsize = batchsize
        self.nz, self.nx, self.ny = self.X.shape
        self.p_shape = p_shape

    def flow(self, count=1024, fixed_random=False):
        if fixed_random:
            np.random.rand(4)
        for i in range(count):
            z_s = np.random.random_integers(0,self.nz-1,self.batchsize)
            x_s = np.random.random_integers(0,self.nx-self.p_shape[0],self.batchsize)
            y_s = np.random.random_integers(0,self.ny-self.p_shape[1],self.batchsize)
            # px,py = [],[]
            # for x,y,z in zip(x_s,y_s,z_s):
            #     px.append(self.X[z,x:x+self.p_shape[0],y:y+self.p_shape[1]])
            #     py.append(self.y[z,x:x+self.p_shape[0],y:y+self.p_shape[1]])
            # patch_X, patch_y = np.array(px), np.array(py)
            patch_X = np.array([self.X[z,x:x+self.p_shape[0],y:y+self.p_shape[1]] for x,y,z in zip(x_s,y_s,z_s)])
            patch_y = np.array([self.y[z,x:x+self.p_shape[0],y:y+self.p_shape[1]] for x,y,z in zip(x_s,y_s,z_s)])
            yield i, patch_X, patch_y

# This is a first version, using 2D convolution.
def create_encoding_layers():
    return [
        # In case here throws an error because the patch is too big, might have to change input_shape (is the same of p_shape)
        Convolution2D(64, 7, 7, input_shape=(1,32,32), border_mode='valid', init='he_normal'), PReLU(),
        MaxPooling2D(pool_size=(2, 2)),
        Convolution2D(64, 5, 5, border_mode='valid', init='he_normal'), PReLU(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(), PReLU(),
        Dense(500, init='he_normal'), PReLU(),
    ]
def create_decoding_layers():
    return[
        Dense(32*7*7, init='he_normal'), PReLU(),
        Reshape((32, 7, 7)),
        UpSampling2D(size=(2,2)),
        ZeroPadding2D(padding=(1,1)),
        Convolution2D(32, 5, 5, border_mode='same', init='he_normal'), PReLU(),
        UpSampling2D(size=(2,2)),
        #ZeroPadding2D(padding=(2,2)),
        Convolution2D(1, 7, 7, border_mode='same', init='he_normal'), PReLU(),
        Activation('sigmoid')
    ]

def main(args):
    # The structure here is predefined
    # Modules from keras are imported here to give a nice clean parser
    # Is it better to put the activation inside the dense layer? Documentation says no.
    patch_size=32
    model = Sequential()
    encoding_layers = create_encoding_layers()
    decoding_layers = create_decoding_layers()
    for i in encoding_layers:
        model.add(i)
    for i in decoding_layers:
        model.add(i)
    
    print("Model defined. Compiling.")

    model.compile(loss="binary_crossentropy", optimizer="adadelta")


    #h5file = tables.open_file(args.datafile, "r")
    #X_train = h5file.root.X.read()
    #y_train = h5file.root.y.read()
    #h5file.close()
    
    # The datafile must be made with the full substack, not make_sup_dataset.py
    cell_data = CellularData(args.datafile)

    print("Model compiled. Dataset read. Training.")
    if args.ID is not None:
        try:
            ba.send_message(args.ID, "Starting training")
        except:
            print("Could not send message!")

    nb_epoch = 20
    pint = 20  # print interval (in minibatches)
    nb_minibatches = 2500
    for e in range(nb_epoch):
        losses, accs = [], []
        for i, X_batch, y_batch in cell_data.flow(count=nb_minibatches, fixed_random=False):
            X_batch = X_batch.reshape(X_batch.shape[0], 1, patch_size, patch_size)
            y_batch = y_batch.reshape(y_batch.shape[0], 1, patch_size, patch_size)
            loss,acc = model.train_on_batch(X_batch, y_batch, accuracy=True)
            losses.append(loss)
            accs.append(acc)
            if (i > 0) and (i % pint) == 0:
                L = np.array(losses)
                A = np.array(accs)
                timestr = time.strftime("%b %d %H:%M:%S", time.localtime())
                print('{:4d}/{:5d} - Loss: global {:.4f}, last set {:.4f} - Acc: global {:.4f}, last set {:.4f} - {}'.format(e, i, L.mean(), L[-pint:].mean(), A.mean(), A[-pint:].mean(), timestr))
        L = np.array(losses)
        A = np.array(accs)
        #Should I save the weights at every epoch?
        print('End of epoch {:4d} - {:.4f} - Acc: {:.4f}'.format(e, L.mean(), A.mean()))
 
    print("Training is completed!")
    if args.ID is not None:
        try:
            ba.send_message(args.ID, "Training is completed!")
        except:
            print("Could not send update message")
    json_string=model.to_json()
    if not os.path.exists(args.destfile):
        os.makedirs(args.destfile)
    open(args.destfile + '/architecture.json', 'w').write(json_string)
    model.save_weights(args.destfile + '/weights.h5', overwrite=True)

    print("Model saved to " + args.destfile)
    
    if args.ID is not None:
        try:
            ba.send_message(args.ID, "Model saved to " + args.destfile)
        except:
            print("Could not send update message")

def get_parser():
    parser = argparse.ArgumentParser(description="""Generates a neural net, trains it on a dataset, then saves the model""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('datafile', type=str,
                        help='The training set generated by make_sup_dataset.py')
    parser.add_argument('destfile', type=str,
                        help='Identifier for the saved model (without file extension)')
    parser.add_argument('-v', '--verbose', dest = 'verb', type=int, default=1, help='0 for no log, 1 for progress bar, 2 for log each epoch')
    parser.add_argument('-t', '--telegram', dest='ID', type=str, default=None, help='addressbook ID for getting updates via Telegram')
    return parser

if __name__ == '__main__':
    parser=get_parser()
    args=parser.parse_args()
    main(args)
