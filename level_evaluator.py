#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
NOTE TO SELF: using 50FPS instead, every frame is 2 centiseconds, which makes retrieval easier than it would be using 2.5 centiseconds.
It is utterly unimportant that the FPS divide the substack's height. Else, 25FPS is a bit bigger but might reduce quality loss. try it.
'''

import numpy as np
from moviepy.editor import *
import timeit
import tables
import argparse

from bcfind import simpleSubStack
from bcfind import mscd
from bcfind.semadec import imtensor
from bcfind.semadec import deconvolver

# Necessary for keeping the right deconvolution size
def pad(tensor, margin):
    x = np.shape(tensor)[0]
    y = np.shape(tensor)[1]
    z = np.shape(tensor)[2]
    output = np.zeros((x+(2*margin), y+(2*margin), z+(2*margin)))
    output[margin:margin+x, margin:margin+y, margin:margin+z] = tensor
    return output

def codify_z(z, fps):
	#Assuming 25 FPS
	start_point = [0, 0, 0, z*(100/fps)]
	if start_point[3] >= 100:
		start_point[2]=start_point[3]/100
		start_point[3]=start_point[3]%100
	if start_point[2] >= 60:
		start_point[1]=start_point[2]/60
		start_point[2]=start_point[2]%60
	if start_point[1] >= 60:
		start_point[0]=start_point[1]/60
		start_point[1]=start_point[1]%60
	start_point = str(start_point[0]).zfill(2) + ":" + str(start_point[1]).zfill(2) + ":" + str(start_point[2]).zfill(2) + "." + str(start_point[3]).zfill(2)
	return start_point

def extract_level(videofile, z, dz, fps):
	mainclip = VideoFileClip(videofile)
	out = []
	for i in range(dz):
		print("Frame " + str(i+1) + " out of " + str(dz))
		frame = mainclip.get_frame(codify_z(z+i, fps))
		frame = frame[:,:,1]
		out.append(frame)
	output = np.array(out)
	return output

def getCoords():
    validInput = False
    print ("X and Y are the starting coordinates of your volume. dX and dY are, respectively, width and height." +
           " This is not counting the extramargin.")
    while (validInput == False):
        x = input("Insert the X value: ")
        if type(x) is int and x >= 0:
            validInput = True
        else:
            print("Please insert an integer greater than or equal to 0.")
    validInput = False
    while (validInput == False):
        y = input("Insert the Y value: ")
        if type(y) is int:
            validInput = True
        else:
            print("Please insert an integer greater than or equal to 0.")
    validInput = False
    while (validInput == False):
        dx = input("Insert the dX value: ")
        if type(dx) is int:
            validInput = True
        else:
            print("Please insert an integer greater than or equal to 0.")
    validInput = False
    while (validInput == False):
        dy = input("Insert the dY value: ")
        if type(dy) is int:
            validInput = True
        else:
            print("Please insert an integer greater than or equal to 0.")
    return x, y, dx, dy

# TODO: This shouldn't be hardcoded, should go in by input
def main(args):
    # The data for the substack proper are readable from the info.json (or .plist)
    z = args.z
    dz = args.dz
    maxX = 3662
    maxY = 8249
    maxZ = 3646

    start_timer = timeit.default_timer()
    print('Extracting subtensor')
    
    if (z+dz > maxZ):
        print("Z coordinate is out of bound for target file! Exiting.")
        return
    
    big_tensor = extract_level(args.infile, z, dz + args.extramargin, args.fps)
    
    # Now for an interactive loop to help catch volumes
    quit = False
    while (quit == False):
        
        #x, y, dx, dy = getCoords()
        x, y, dx, dy = 964, 256, 275, 239

        if (x + dx > maxX or y + dy > maxY):
            print("Coordinates are out of bound for target file! Exiting.")
            return
        
        np_tensor_3d = big_tensor[:, y:y+dy+args.extramargin, x:x+dx+args.extramargin]
        
        print('Reading standardization data from ' + args.trainfile)
        h5 = tables.openFile(args.trainfile)
        Xmean = h5.root.Xmean[:].astype(np.float32)
        Xstd = h5.root.Xstd[:].astype(np.float32)
        h5.close()
        
        extract_timer = timeit.default_timer()
        print("Extraction time: " + str(extract_timer - start_timer))
        
        print('Starting semantic devonvolution of a (' + str(dx+args.extramargin) + ',' + str(dy+args.extramargin) + ',' + str(dz+args.extramargin) +
              ') volume starting at (' + str(x) + ',' + str(y) + ',' + str(z) + ')')
        # Importing here to have a clean --help
        from keras.models import model_from_json
        
        net_model = model_from_json(open(args.model + '/architecture.json').read())
        net_model.load_weights(args.model + '/weights.h5')
        
        reconstruction = deconvolver.filter_volume(pad(np_tensor_3d, 6), Xmean, Xstd, args.extramargin, net_model, 4, do_cython=args.do_cython, trainfile=args.trainfile)
        
        reconstruct_timer = timeit.default_timer()
        print("Reconstruction time: " + str(reconstruct_timer - extract_timer))
        
        volume = simpleSubStack.simpleSubStack()
        volume.load_volume(reconstruction)
        # save_image requires the full SubStack item, not my rough copy
        args.save_image = False
        # Without changing mscd.ms, right now, there is no way of controlling the file it goes to write.
        # at each iteration it would overwrite the old ms.marker
        mscd.ms(volume, args)
        meanshift_timer = timeit.default_timer()
        print("Mean Shift time: " + str(meanshift_timer - reconstruct_timer))
        print("Total time: " + str(meanshift_timer - start_timer))
        
        cont = raw_input("Do you want to get another volume in the same z-dz range? [y]/n: ")
        if (cont == 'n' or cont == 'no'):
            quit = True
    
def get_parser():
    parser = argparse.ArgumentParser(description="""
    Runs a semantic deconvolution - mean shift combination on a user-defined volume in a mp4-compressed file.
    25 FPS recommended for best results.
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('infile', metavar='infile', type=str,
                        help='The mp4 file to convert into array')
    parser.add_argument('model', metavar='model', type=str,
                        help='The folder where the Keras neural network was saved')
    parser.add_argument('trainfile', metavar='trainfile', type=str,
                        help='The hdf5 dataset the neural network was trained on')
    parser.add_argument('z', metavar='z', type=int,
                        help='The z coordinate of the frames to extract')
    parser.add_argument('dz', metavar='dz', type=int,
                        help='The number of extracted frames')
    parser.add_argument('outdir', metavar='outdir', type=str,
                        help='The directory where to save the results')
    
    parser.add_argument('--extramargin', metavar='extramargin', dest='extramargin',
                        action='store', type=int, default=6,
                        help='Extra margin for convolution')
    parser.add_argument('--fps', metavar='fps', dest='fps',
                        action='store', type=int, default=25,
                        help='Frames per second of the mp4')
    parser.add_argument('-r', '--hi_local_max_radius', metavar='r', dest='hi_local_max_radius',
                        action='store', type=float, default=6,
                        help='Radius of the seed selection ball (r)')
    parser.add_argument('-t', '--seeds_filtering_mode', dest='seeds_filtering_mode',
                        action='store', type=str, default='soft',
                        help="Type of seed selection ball ('hard' or 'soft')")
    parser.add_argument('-R', '--mean_shift_bandwidth', metavar='R', dest='mean_shift_bandwidth',
                        action='store', type=float, default=5.5,
                        help='Radius of the mean shift kernel (R)')
    parser.add_argument('-f', '--floating_point', dest='floating_point', action='store_true',
                        help='If true, cell centers are saved in floating point.')
    parser.add_argument('-m', '--min_second_threshold', metavar='min_second_threshold', dest='min_second_threshold',
                        action='store', type=int, default=15,
                        help="""If the foreground (second threshold in multi-Kapur) is below this value
                        then the substack is too dark and assumed to contain no soma""")
    parser.add_argument('-M', '--max_expected_cells', metavar='max_expected_cells', dest='max_expected_cells',
                        action='store', type=int, default=10000,
                        help="""Max number of cells that may appear in a substack""")
    
    parser.add_argument('--do_cython', dest='do_cython', action='store_true', help='use the compiled cython modules in deconvolver.py')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
