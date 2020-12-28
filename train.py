import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from unet3d import UNet3D
from loss import combined_loss
import time
import numpy as np
from torch.utils.data import DataLoader
from metrics import (get_metrics_from_arrays,
                     get_metrics_str, get_metrics,
                     get_metric_csv_row,
                     get_metric_header_str)
from datasets import create_datasets


def train_network(train_ds, epochs, log_path):
    """ Train a network for {epochs}
        using data from {train_ds}.
        Retrun metrics at the end of each epoch.
    """
    dataloader = DataLoader(train_ds, batch_size=10, 
                            shuffle=True, num_workers=12)
    cnn = UNet3D(im_channels=1, out_channels=2).cuda()
    cnn = nn.DataParallel(cnn)
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01,
                                momentum=0.9, nesterov=True)
    loss_fn = combined_loss
    train_log = open(log_path, 'w+')
    print(get_metric_header_str(), file=train_log)
    metric_list = []
    train_start = time.time()
    print('start training epochs')
    for i in range(epochs):
        start = time.time()
        print('start epoch', i)
        tps = []
        tns = []
        fps = []
        fns = []
        for step, (image_batch, labels_batch) in enumerate(dataloader):
            print('step', step, end='\r')
            optimizer.zero_grad()
            outputs = cnn(image_batch.cuda())
            labels_batch  = labels_batch.cuda().long()
            loss = loss_fn(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            preds_int = torch.argmax(outputs, 1)
            foregrounds_int = labels_batch
            tps.append(torch.sum((foregrounds_int == 1) *
                                 (preds_int == 1)).cpu().numpy())
            tns.append(torch.sum((foregrounds_int == 0) *
                                 (preds_int == 0)).cpu().numpy())
            fps.append(torch.sum((foregrounds_int == 0) *
                                 (preds_int == 1)).cpu().numpy())
            fns.append(torch.sum((foregrounds_int == 1) *
                                 (preds_int == 0)).cpu().numpy())
        m = get_metrics(np.sum(tps), np.sum(fps), np.sum(tns), np.sum(fns))
        print('epoch', i, get_metrics_str(m),
              'duration', round(time.time() - start, 2))
        print(f'{get_metric_csv_row(m, train_start)}', file=train_log)
        metric_list.append(m)
    return metric_list


if __name__ == '__main__':
    data_dir = os.path.join('data', 'ThoracicOAR_quarter')
    train_ds, val_ds, test_ds = create_datasets(data_dir)
    total_logs = len(os.listdir('logs'))
    log_csv_path = f'logs/train_metrics_combined_{total_logs}.csv'
    metric_list = train_network(train_ds, 400, log_csv_path)
    print('metric_list', metric_list)
    print('metric_list', [m['dice'] for m in metric_list])
