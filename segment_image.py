"""
Copyright (C) 2020 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from unet3d import UNet3D
import numpy as np


def get_coords_3d(annot_shape, out_tile_shape):
    """ Get the coordinates relative to the output image for the 
        validation routine. These coordinates will lead to patches
        which cover the image with minimum overlap (assuming fixed size patch) """

    assert len(annot_shape) == 3, str(annot_shape) # d, h, w
    
    depth_count = math.ceil(annot_shape[0] / out_tile_shape[0])
    vertical_count = math.ceil(annot_shape[1] / out_tile_shape[1])
    horizontal_count = math.ceil(annot_shape[2] / out_tile_shape[2])

    # first split the image based on the tiles that fit
    z_coords = [d*out_tile_shape[0] for d in range(depth_count-1)] # z is depth
    y_coords = [v*out_tile_shape[1] for v in range(vertical_count-1)]
    x_coords = [h*out_tile_shape[2] for h in range(horizontal_count-1)]

    # The last row and column of tiles might not fit
    # (Might go outside the image)
    # so get the tile positiion by subtracting tile size from the
    # edge of the image.
    lower_z = annot_shape[0] - out_tile_shape[0]
    bottom_y = annot_shape[1] - out_tile_shape[1]
    right_x = annot_shape[2] - out_tile_shape[2]

    z_coords.append(max(0, lower_z))
    y_coords.append(max(0, bottom_y))
    x_coords.append(max(0, right_x))

    # because its a cuboid get all combinations of x, y and z
    tile_coords = [(x, y, z) for x in x_coords for y in y_coords for z in z_coords]
    return tile_coords


def segment(cnn, image, batch_size, in_tile_shape, out_tile_shape):
    """
    in_tile_shape and out_tile_shape are (depth, height, width)
    """
    # Return prediction for each pixel in the image
    # The cnn will give a the output as channels where
    # each channel corresponds to a specific class 'probability'
    # don't need channel dimension
    # make sure the width, height and depth is at least as big as the tile.
    assert len(image.shape) == 3, str(image.shape)
    assert image.shape[0] >= in_tile_shape[0], f"{image.shape[0]},{in_tile_shape[0]}"
    assert image.shape[1] >= in_tile_shape[1], f"{image.shape[1]},{in_tile_shape[1]}"
    assert image.shape[2] >= in_tile_shape[2], f"{image.shape[2]},{in_tile_shape[2]}"

    depth_diff = in_tile_shape[0] - out_tile_shape[0]
    width_diff = in_tile_shape[1] - out_tile_shape[1]
    
    out_im_shape = (image.shape[0] - depth_diff,
                    image.shape[1] - width_diff,
                    image.shape[2] - width_diff)

    coords = get_coords_3d(out_im_shape, out_tile_shape)
    coord_idx = 0
    # segmentation for the full image
    # assign once we get number of classes from the cnn output shape.
    seg = np.zeros(out_im_shape, dtype=np.int8)
    while coord_idx < len(coords):
        tiles_to_process = []
        coords_to_process = []
        for _ in range(batch_size):
            if coord_idx < len(coords):
                coord = coords[coord_idx]
                x_coord, y_coord, z_coord = coord
                tile = image[z_coord:z_coord+in_tile_shape[0],
                             y_coord:y_coord+in_tile_shape[1],
                             x_coord:x_coord+in_tile_shape[2]]
                # need to add channel dimension for GPU processing.
                tile = np.expand_dims(tile, axis=0)
                assert tile.shape[1] == in_tile_shape[0], str(tile.shape)
                assert tile.shape[2] == in_tile_shape[1], str(tile.shape)
                assert tile.shape[3] == in_tile_shape[2], str(tile.shape)
                coord_idx += 1
                tiles_to_process.append(tile) # need channel dimension
                coords_to_process.append(coord)

        tiles_to_process = np.array(tiles_to_process)
        tiles_for_gpu = torch.from_numpy(tiles_to_process)
        tiles_for_gpu = tiles_for_gpu.cuda()
        tile_predictions = cnn(tiles_for_gpu)
        tile_predictions = softmax(tile_predictions, 1)[:, 1, :] # just foreground
        tile_predictions = (tile_predictions > 0.5).type(torch.int8)
        pred_np = tile_predictions.data.cpu().numpy()
        out_tiles = pred_np.reshape(([len(tiles_for_gpu)] + list(out_tile_shape)))
        # add the predictions from the gpu to the output segmentation
        # use their correspond coordinates
        for tile, (x_coord, y_coord, z_coord) in zip(out_tiles, coords_to_process):
            seg[z_coord:z_coord+tile.shape[0],
                y_coord:y_coord+tile.shape[1],
                x_coord:x_coord+tile.shape[2]] = tile
    return seg
