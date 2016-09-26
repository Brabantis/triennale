#!/usr/bin/env python
"""
Generate a neural network from a dataset
"""
# Make sure that theano.config.openmp is set to True for multiprocessor elaboration
import tables
import numpy
import argparse
import os

from bcfind.util import bot_alert as ba

# 2197 is the number of points in a training patch
def main(args):
    # The structure here is predefined
    # Modules from keras are imported here to give a nice clean parser

    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Reshape
    from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
    from keras.layers.advanced_activations import PReLU
    from keras.layers.normalization import BatchNormalization
    # Is it better to put the activation inside the dense layer? Documentation says no.
    model = Sequential()
    model.add(Reshape((1, 13, 13, 13), input_shape=(2197,)))
    model.add(Convolution3D(4, 5, 5, 5, border_mode="same", init="he_normal"))
    model.add(PReLU())
    model.add(Convolution3D(4, 3, 3, 3, border_mode="same", init="he_normal"))
    model.add(PReLU())
    model.add(Flatten())
    model.add(Dense(output_dim=2197, init="he_normal"))
    model.add(Activation("sigmoid"))

    print("Model defined. Compiling.")

    model.compile(loss="binary_crossentropy", optimizer="adadelta")

    h5file = tables.open_file(args.datafile, "r")
    X_train = h5file.root.X.read()
    y_train = h5file.root.y.read()
    h5file.close()

    print("Model compiled. Dataset read. Training.")
    if args.ID is not None:
        try:
            ba.send_message(args.ID, "Starting training")
        except:
            print("Could not send message!")

    # Likely too few epochs, but the loss diminishes very slowly.
    model.fit(X_train, y_train, nb_epoch=32, batch_size=64, verbose=args.verb) 
    """
    # I did something wrong here, that caused precision and recall to drop.
    losses, accs = [], []
    batch_size = 64
    num_epochs = 32
    for epoch in range(num_epochs):
        print("Starting epoch " + str(epoch+1))
        for count in range(0, len(X_train), 64):
            X_batch = X_train[count : count + batch_size]
            y_batch = y_train[count : count + batch_size]
            loss, acc = model.train_on_batch(X_batch, y_batch, accuracy=True)
            losses.append(loss)
            accs.append(acc)
        L=numpy.array(losses)
        A=numpy.array(accs)
        print("End of epoch " + str(epoch+1) + " - Loss: " + str(L.mean()) + ", Acc: " + str(A.mean()))
        if not os.path.exists(args.destfile):
            os.makedirs(args.destfile)
        model.save_weights(args.destfile + '/weights.h5', overwrite=True)
    """
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