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


def load_csv(csv_path, header_names, types):
    """
    Example usage:
    file_names, order = load_csv('output_csv/file_order.csv',
                                 ['file_name', 'order'],
                                 [str, int])
    """
    lines = open(csv_path).readlines()
    headers = lines[0].strip().split(',')
    for h in header_names:
        assert h in headers, f'did not find {h} in {headers}'
    header_values = [[] for h in header_names]
    header_idxs = [headers.index(h) for h in header_names]
    for l in lines[1:]:
        parts = l.strip().split(',')
        for i in range(len(header_idxs)):
            header_values[i].append(types[i]((parts[header_idxs[i]])))
    return header_values

