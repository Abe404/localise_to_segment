"""
Computing metrics on output segmentations 

Copyright (C) 2019, 2020 Abraham George Smith

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

# pylint: disable=C0111,R0913
from dataclasses import dataclass
from datetime import datetime
import time
import numpy as np


metric_headers = ['seconds', 'time', 'tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'dice']


def get_metric_header_str():
    return  ','.join(metric_headers)

def compute_metrics_from_binary_masks(seg, gt):
    gt = gt.reshape(-1).astype(int)
    seg = seg.reshape(-1).astype(int)
    assert len(gt) == len(seg)
    return Metrics(
        tp=(np.sum((gt == 1) * (seg == 1))),
        tn=(np.sum((gt == 0) * (seg == 0))),
        fp=(np.sum((gt == 0) * (seg == 1))),
        fn=(np.sum((gt == 1) * (seg == 0)))
    )



@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    
    def total(self):
        return self.tp + self.tn + self.fp + self.fn
    
    def accuracy(self):
        return (self.tp + self.tn) / self.total()

    def precision(self):
        if self.tp > 0:
            return self.tp / (self.tp + self.fp)
        return float('NaN')

    def recall(self): 
        if self.tp > 0:
            return self.tp / (self.tp + self.fn)
        return float('NaN')

    def dice(self): 
        if self.tp > 0:
            return 2 * ((self.precision() * self.recall()) / (self.precision() + self.recall()))
        return float('NaN')
    
    def true_mean(self):
        return (self.tp + self.fn) / self.total()

    def total_true(self):
        return (self.tp + self.fn),

    def total_pred(self):
        return (self.fp + self.tp)

    def __add__(self, other):
        return Metrics(tp=self.tp+other.tp, 
                       fp=self.fp+other.fp, 
                       tn=self.tn+other.tn, 
                       fn=self.fn+other.fn)
    
    def __str__(self, to_use=None):
        out_str = ""
        for name in metric_headers:
            if to_use is None or name in to_use:
                if hasattr(self, name):
                    val = getattr(self, name)
                    if callable(val):
                        val = val()
                    out_str += f" {name} {val:.4g}"
        return out_str

    def csv_row(self, start_time):
        now_str = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        seconds = time.time() - start_time
        parts = [seconds, now_str, self.tp,
                 self.fp, self.tn, self.fn,
                 round(self.precision(), 4), round(self.recall(), 4),
                 round(self.dice(), 4)]
        return ','.join([str(p) for p in parts])

