#!/usr/bin/env python
"""
Script that creates a training set for semantic deconvolution.
"""

from __future__ import print_function
import argparse
# Is this still necessary? Don't think so
import cPickle as pickle
import tables
import numpy as np
import timeit
import re
import sys


from bcfind.volume import SubStack
from bcfind.semadec import imtensor
from bcfind.semadec import deconvolver

def pad(tensor, margin):
    x = np.shape(tensor)[0]
    y = np.shape(tensor)[1]
    z = np.shape(tensor)[2]
    output = np.zeros((x+(2*margin), y+(2*margin), z+(2*margin)))
    output[margin:margin+x, margin:margin+y, margin:margin+z] = tensor
    return output

def main(args):

    total_start = timeit.default_timer()
    print('Starting reconstruction of volume %s ...'%(args.substack_id))

    substack = SubStack(args.indir,args.substack_id)
    substack.load_volume()
    tensor = substack.get_volume()

    # Changing the tensor so that it has a 6 pixel black padding. Hopefully it won't frick things up too much. Else, mail time.
    print("The shape of the tensor before padding: " + str(np.shape(tensor)))
    tensor = pad(tensor, 6)
    print("The shape of the tensor after padding: " + str(np.shape(tensor)))

    if not args.local_mean_std:
        print('Reading standardization data from', args.trainfile)
        h5 = tables.openFile(args.trainfile)
        Xmean = h5.root.Xmean[:].astype(np.float32)
        Xstd = h5.root.Xstd[:].astype(np.float32)
        h5.close()
    else:
        Xmean=None
        Xstd=None
    
    print('Starting semantic devonvolution of volume', args.substack_id)
    # Importing here to have a clean --help
    from keras.models import model_from_json
    model = model_from_json(open(args.model + '/architecture.json').read())
    model.load_weights(args.model + '/weights.h5')
    
    minz = int(re.split('[a-zA-z0-9]*_',substack.info['Files'][0])[1].split('.tif')[0])
    # Remove the margin, I have changed deconvolver to use a fized number instead of the extramargin. Hope it works.
    reconstruction = deconvolver.filter_volume(tensor, Xmean, Xstd,
                                               args.extramargin, model, args.speedup, do_cython=args.do_cython, trainfile=args.trainfile)
    imtensor.save_tensor_as_tif(reconstruction, args.outdir+'/'+args.substack_id, minz)

    print ("total time reconstruction: %s" %(str(timeit.default_timer() - total_start)))


def get_parser():
    parser = argparse.ArgumentParser(description="""
    Preprocess a substack using a neural network model
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('indir', metavar='indir', type=str,
                        help='needs indir/info.json, substacks, e.g. indir/000, and ground truth, e.g. indir/000-GT.marker')
    parser.add_argument('substack_id', metavar='substack_id', type=str,
                        help='substack identifier, e.g. 010608')
    # parser.add_argument('tensorimage', metavar='tensorimage', type=str,
    #                    help='path to the tensor image .h5 file')
    parser.add_argument('model', metavar='model', type=str,
                        help='folder containing a trained Keras network, saved as architecture.json and weights.h5')
    parser.add_argument('trainfile', metavar='trainfile', type=str,
                        help='HDF5 file on which the network was trained (should contain mean/std arrays)')
    parser.add_argument('outdir', metavar='outdir', type=str,
                        help='where preprocessed volume will be saved')
    parser.add_argument('--extramargin', metavar='extramargin', dest='extramargin',
                        action='store', type=int, default=6,
                        help='Extra margin for convolution. Temporarily set to 0')
    parser.add_argument('--speedup', metavar='speedup', dest='speedup',
                        action='store', type=int, default=4,
                        help='convolution stride (isotropic along X,Y,Z)')
    parser.add_argument('--local_mean_std', dest='local_mean_std', action='store_true',
                        help='computcompute mean and std locally from the substack')
    parser.add_argument('--do_cython', dest='do_cython', action='store_true', help='use the compiled cython modules in deconvolver.py')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

