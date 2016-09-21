#!/usr/bin/env python
"""
Creates a training set for semantic deconvolution through a convolutional network.
"""

from __future__ import print_function
import numpy as np
import tables
import argparse
import progressbar as pb

from bcfind import volume
from scipy.spatial import cKDTree
from bcfind.semadec import imtensor
import scipy.ndimage.filters as gfilter



def inside_margin(c, substack):
    """ Are we inside the safe region?"""
    m = substack.plist['Margin']/2
    return min(c.x-m,c.y-m,c.z-m,substack.info['Width']-m-c.x,substack.info['Height']-m-c.y,substack.info['Depth']-m-c.z)



def make_dataset(tensorimage, ss, C, L=12, size=None, save_tiff_files=False, negatives=False, margin=None):
    hf5 = tables.openFile(tensorimage, 'r')
    X0,Y0,Z0 = ss.info['X0'], ss.info['Y0'], ss.info['Z0']
    origin = (Z0, Y0, X0)
    H,W,D = ss.info['Height'], ss.info['Width'], ss.info['Depth']
    ss_shape = (D,H,W)
    print('Loading data for substack', ss.substack_id)
    np_tensor_3d = hf5.root.full_image[origin[0]:origin[0]+ss_shape[0],
                                       origin[1]:origin[1]+ss_shape[1],
                                       origin[2]:origin[2]+ss_shape[2]]
    print("As loaded, min, max and average of X are " + str(np.min(np_tensor_3d)) + ", " + str(np.max(np_tensor_3d)) + ", " + str(np.average(np_tensor_3d)))
    X = []
    y = []
    patchlen = (1+2*size)**3
    print('Preparing..')
    kdtree = cKDTree(np.array([[c.x,c.y,c.z] for c in C]))
    nrej_intensity = 0
    nrej_near = 0
    target_tensor_3d = np.zeros(np_tensor_3d.shape)
    for c in C:
        if inside_margin(c,ss) > 0:
            target_tensor_3d[c.z, c.y, c.x] = 1
    target_tensor_3d = gfilter.gaussian_filter(target_tensor_3d, sigma=3.5,
                                               mode='constant', cval=0.0,
                                               truncate=1.5)
    # undo scipy normalization of the gaussian filter
    target_tensor_3d = (target_tensor_3d / np.max(target_tensor_3d))
    hf5.close()

    X = np_tensor_3d
    y = target_tensor_3d

    return X, y

def main(args):
    data = []
    target = []
    for substack_id in args.substack_ids:
        substack = volume.SubStack(args.indir,substack_id)
        gt_markers = args.indir+'/'+substack_id+'-GT.marker'
        print('Loading ground truth markers from',gt_markers)
        C = substack.load_markers(gt_markers,from_vaa3d=True)
        for c in C:
            c.x -= 1
            c.y -= 1
            c.z -= 1
        sdata,starget = make_dataset(args.tensorimage, substack, C, size=args.size, negatives=args.negatives, margin=args.margin)

        data.extend(sdata)
        target.extend(starget)

    X = np.zeros((len(data),data[0].shape[0], data[0].shape[1]), dtype=np.float32)
    y = np.zeros((len(data),data[0].shape[0], data[0].shape[1]), dtype=np.float32)
    pbar = pb.ProgressBar(widgets=['Converting to 32-bit numpy array %d examples: ' % X.shape[0], pb.Percentage()],
                          maxval=X.shape[0]).start()
    for i in range(X.shape[0]):
        X[i] = (data[i]/255.0).astype(np.float32)
        #X[i] = (data[i]).astype(np.float32)
        y[i] = target[i].astype(np.float32)
        pbar.update(i+1)
    pbar.finish()

    print('Data set shape:', X.shape, 'size:', X.nbytes/(1024*1024), 'MBytes')
    print('target shape:', y.shape, 'size:', y.nbytes/(1024*1024), 'MBytes')

    print('Standardizing')

    print("Before standardisation, min, max and average of X are " + str(np.min(X)) + ", " + str(np.max(X)) + ", " + str(np.average(X)))

    Xmean = X.mean(axis=0)
    print(Xmean)
    Xstd = X.std(axis=0)
    print(Xstd)
    X = (X - Xmean) / Xstd

    print("After standardisation, min, max and average of X are " + str(np.min(X)) + ", " + str(np.max(X)) + ", " + str(np.average(X)))


    #The standardisation gives back a dataset with a certain range of values. Multiplying it by 255 as it is done for saving will make it way too bright. Now check max, min and avg of the predicted
    print('Saving training data to',args.outfile)
    h5file = tables.openFile(args.outfile, mode='w', title="Training set")
    root = h5file.root
    h5file.createArray(root, "X", X)
    h5file.createArray(root, "y", y)
    h5file.createArray(root, "Xmean", Xmean)
    h5file.createArray(root, "Xstd", Xstd)
    h5file.close()


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('indir', metavar='indir', type=str,
                        help='needs indir/info.json, substacks, e.g. indir/000, and ground truth, e.g. indir/000-GT.marker')
    parser.add_argument('tensorimage', metavar='tensorimage', type=str,
                        help='HDF5 file containing the whole stack')
    parser.add_argument('outfile', metavar='outfile', type=str,
                        help='Name of the HDF5 file where results will be saved')
    parser.add_argument('substack_ids', metavar='substack_ids', type=str, nargs='+',
                        help='substacks identifier, e.g. 010608')
    parser.add_argument('-s', '--size', dest='size',
                        action='store', type=int, default=6,
                        help='Input and output patches are cubes of side (2*size+1)**3')
    parser.add_argument('-m', '--margin', dest='margin', type=int, default=40, help='Overlap between adjacent substacks')
    parser.add_argument('--negatives', dest='negatives', action='store_true', help='include "negative" (non cell) examples.')
    parser.add_argument('--no-negatives', dest='negatives', action='store_false', help='Include only cell examples.')
    parser.set_defaults(negatives=False)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
