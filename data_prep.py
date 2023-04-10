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

import argparse
from raw_dataset import RawDataset


def prep_data(im_dir, annot_dir, organ_padding, patch_width, patch_depth):
    patch_size = (patch_depth, patch_width, patch_width)
    ds = RawDataset(im_dir, annot_dir, patch_size, organ_padding)
    ds.create_cropped_dataset()
    ds.create_low_res_dataset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument('--imdir', type=str, required=True, help='path to full image directory')
    parser.add_argument('--annotdir', type=str, required=True, help='path to full annotation directory')
    parser.add_argument('--patchwidth', type=int, default=256,
                        help='input patch width for network (or for cropping)')
    parser.add_argument('--patchdepth', type=int, default=64,
                        help='input patch depth for network (or for cropping)')
    parser.add_argument('--organpadding', type=int, default=15,
                        help=('how much to pad around the cropped organ.'
                              'Makes sure even organs on the edge of the image will have some '
                              'background context (will be padded to 0)'))
    args, _ = parser.parse_known_args()
    prep_data(args.imdir, args.annotdir, args.organpadding, args.patchwidth, args.patchdepth)
