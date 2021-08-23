import time
import os
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from torch.autograd import Variable

from dataset import HDR_Dataset, HDR_DataLoader_pre
from model import DHDR
from tqdm import trange
import numpy as np

torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch-size', '-N', type=int, default=8, help='batch size')
parser.add_argument(
    '--train', '-f', default="./dataset_select/Train/TrainSequence.h5", type=str, help='folder of training images')
parser.add_argument(
    '--max-epochs', '-e', type=int, default=300, help='max epochs')
parser.add_argument('--crop_size', type=int, default=256, help='image height')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')

parser.add_argument('--nDenselayer', type=int, default=3, help='nDenselayer of RDB')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--filters_in', type=int, default=6, help='number of channels in')
parser.add_argument('--filters_out', type=int, default=3, help='number of channels out')
parser.add_argument('--groups', type=int, default=8, help='number of groups')
args = parser.parse_args()

num = 13109

train_set = HDR_Dataset(datadir=args.train, crop_size=args.crop_size)

train_loader = HDR_DataLoader_pre(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
# train_loader = data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

print('total images: {}; total batches: {}'.format(
    len(train_set), len(train_loader)))

hdr = DHDR(args=args).cuda()
# hdr._initialize_weights()
solver = optim.Adam(
    [
        {
            'params': hdr.parameters()
        },
    ],
    lr=args.lr)

MU = 5000.0


def tonemap(images):  # input/output 0~1
    return torch.log(1.0 + MU * images) / np.log(1.0 + MU)


def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    hdr.load_state_dict(
        torch.load('checkpoint/hdr_{}_{:08d}.pth'.format(s, epoch)))


def save(index, epoch=True):
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'

    torch.save(hdr.state_dict(), 'checkpoint/hdr_{}_{:08d}.pth'.format(s, index))


scheduler = LS.MultiStepLR(solver, milestones=[200], gamma=0.5)

args.checkpoint = 135
last_epoch = args.checkpoint
if args.checkpoint:
    resume(args.checkpoint)
    scheduler.last_epoch = last_epoch

for epoch in trange(last_epoch + 1, args.max_epochs + 1):

    scheduler.step()

    for batch, data in enumerate(train_loader):
        batch_t0 = time.time()

        in_imgs = Variable(data[0].type(torch.float32).cuda())
        ref_HDR = Variable(data[1].type(torch.float32).cuda())

        solver.zero_grad()

        bp_t0 = time.time()

        out_HDR = hdr(in_imgs)

        bp_t1 = time.time()

        res = tonemap(ref_HDR) - tonemap(out_HDR)

        loss = res.abs().mean()

        loss.backward()

        solver.step()

        batch_t1 = time.time()

        index = (epoch - 1) * len(train_loader) + batch

        print(
            '[TRAIN]index({}) Epoch[{}]({}/{}); Loss: {:.6f}; Forwordtime: {:.4f} sec; Batch: {:.4f} sec; lr: {:.6f}'.
                format(index, epoch, batch + 1,
                       len(train_loader), loss.data, bp_t1 - bp_t0, batch_t1 - batch_t0, solver.param_groups[0]["lr"]))

        ## save checkpoint every 2000 training steps
        ## if index % 2000 == 0 and index > 400000:
        ##     save(index, False)

    if epoch % 5 == 0:
        save(epoch)
