import os
from utils import *
import torch
from torch.autograd import Variable
import argparse
from model import DHDR
import h5py

torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/hdr.pth')
parser.add_argument('--results_dir', type=str, default='results')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--c_dim', type=int, default=3)
parser.add_argument('--dataset', type=str, default='./dataset_select/Test')
parser.add_argument('--test_h', type=int, default=960)
parser.add_argument('--test_w', type=int, default=1440)

parser.add_argument('--nDenselayer', type=int, default=3, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--filters_in', type=int, default=6, help='number of channels in')
parser.add_argument('--filters_out', type=int, default=3, help='number of channels out')
parser.add_argument('--groups', type=int, default=8, help='number of groups')
args = parser.parse_args()

if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)

MU = 5000. # tunemapping parameter
def tonemap_np(images):  # input/output 0~1
    return np.log(1 + MU * images) / np.log(1 + MU)


GAMMA = 2.2 # LDR&HDR domain transform parameter
def LDR2HDR(img, expo): # input/output 0~1
    return img ** GAMMA / expo


def center_crop(x, image_size):
    crop_h, crop_w = image_size
    _, _, h, w = x.shape
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return x[:, :, max(0,j):min(h,j+crop_h), max(0,i):min(w,i+crop_w)]


def get_input(scene_dir, image_size):
    hf = h5py.File(scene_dir)
    inputs = np.array(hf.get('IN'))
    inputs = center_crop(inputs, image_size)
    ref_HDR = np.array(hf.get('GT'))
    ref_HDR = center_crop(ref_HDR, image_size)
    return inputs, ref_HDR


scene_dirs = sorted(os.listdir(args.dataset))
nScenes = len(scene_dirs)
num_batch = int(np.ceil(nScenes/args.batch_size))

hdr = DHDR(args=args)
hdr = hdr.cuda()
hdr.load_state_dict(torch.load(args.checkpoint_dir, map_location=torch.device('cpu')))
hdr.eval()

for idx in range(nScenes):
    print('batch no. %d:' % (idx + 1))

    input_features, ref_HDR = get_input(os.path.join(args.dataset, scene_dirs[idx]),
                                        [args.test_w, args.test_h])

    with torch.no_grad():
        input_features = Variable(torch.from_numpy(input_features)).cuda()
        out_HDR = hdr(input_features)

    out_HDR = np.transpose(np.squeeze(out_HDR.data.cpu().numpy()), axes=(2, 1, 0))

    ref_HDR = np.transpose(np.squeeze(ref_HDR), axes=(2, 1, 0))
    out_HDR_T = tonemap_np(out_HDR)
    ref_HDR_T = tonemap_np(ref_HDR)

    torch.cuda.empty_cache()

    imsave(out_HDR, os.path.join(args.results_dir, 'test_{:03d}_{:03d}_HDR.hdr'.format(idx, idx + 1)))
    imsave(ref_HDR, os.path.join(args.results_dir, 'test_{:03d}_{:03d}_ref_HDR.hdr'.format(idx, idx + 1)))
    imsave(os.path.join(args.results_dir, 'test_{:03d}_{:03d}_tonemapped.png'.format(idx, idx + 1)), out_HDR_T)
    imsave(os.path.join(args.results_dir, 'test_{:03d}_{:03d}_ref_tonemapped.png'.format(idx, idx + 1)), ref_HDR_T)
