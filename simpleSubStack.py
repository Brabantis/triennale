from __future__ import print_function

import numpy as np
import os
import cPickle as pickle

from scipy.spatial import cKDTree
from PIL import Image

from bcfind import log

SHARE_DIR = os.path.dirname(log.__file__)+'/share'
hi2rgb = pickle.load(open(SHARE_DIR+'/hi2rgb.pickle', 'rb'))

class simpleSubStack():
    
    def load_volume(self, inarray):
        """Loads a sequence of images into a stack

        Parameters
        ----------
        convert_to_gray : bool
            Should be set to true if reading from RGB tiff files
        flip : bool
            Flip along vertical (Y) axis
        ignore_info_files : bool
            If true, don't trust filenames in the info.json file
        """
        self.imgs = []
        self.info = {'Depth': inarray.shape[0], 'Width': inarray.shape[2], 'Height': inarray.shape[1]}
        for z in range(inarray.shape[0]):
            img_z = inarray[z, :, :]
            self.imgs.append(img_z)
        print(str(z+1) + " images read into stack")

    def neighbors_graph(self, C):
        X = np.array([[c.x, c.y, c.z] for c in C])
        kdtree = cKDTree(X)
        for c in C:
            distances, neighbors = kdtree.query([c.x, c.y, c.z], 6)
            c.distances = sorted(distances)[1:]

    def save_markers(self, filename, C, floating_point=False):
        """save_markers(filename, C)

        Save markers to a Vaa3D readable file.

        Parameters
        ----------
        filename : str
            Name of the file where markers are saved
        C : list
            List of :class:`Center` objects
        floating_point: bool
            If true, save coordinates in floating point, else round to int
        """
        if len(C) == 0:  # might happen when logging deletions as marker files
            return
        self.neighbors_graph(C)
        ostream = open(filename, 'w')
        print('##x,y,z,radius,shape,name,comment, color_r,color_g,color_b', file=ostream)
        # for i,c in enumerate(C):
        for c in C:
            r, g, b = hi2rgb[int(255*c.hue)][156]
            radius = 0
            shape = 1
            if floating_point:
                cx, cy, cz = c.x, c.y, c.z
            else:
                cx, cy, cz = int(round(c.x)), int(round(c.y)), int(round(c.z))
            comment = str(c)
            print(','.join(map(str, [1+cx, 1+cy, 1+cz, radius, shape, c.name, comment, r, g, b])), file=ostream)
        ostream.close()
