import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
import cv2
import h5py
from superpoint import SuperPointFrontend
import torchvision.transforms as transforms
import torch.nn as nn
import torch

def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='extract sift.')
parser.add_argument('--input_path', type=str, default='../raw_data/yfcc100m/',
  help='Image directory or movie file or "camera" (for webcam).')
parser.add_argument('--img_glob', type=str, default='*/*/images/*.jpg',
  help='Glob match if directory of images is specified (default: \'*/images/*.jpg\').')
parser.add_argument('--num_kp', type=int, default='2000',
  help='keypoint number, default:2000')
parser.add_argument('--suffix', type=str, default='sift-2000',
  help='suffix of filename, default:sift-2000')


class ExtractSuper(object):
    def __init__(self):
        self.fe = SuperPointFrontend(weights_path='./superpoint_v1.pth', nms_dist=2,  conf_thresh=0.001, nn_thresh=0.7, cuda=True)

    def run(self, img_path):
        grayim = cv2.imread(img_path, 0)
        grayim = (grayim.astype('float32') / 255.)

        pts, desc, heatmap = self.fe.run(grayim)
        desc = desc.T
        pts = pts.T

        return pts, desc

class ExtractSIFT(object):
    def __init__(self, num_kp, contrastThreshold=1e-5):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)

    def run(self, img_path):
        img = cv2.imread(img_path)
        cv_kp, desc = self.sift.detectAndCompute(img, None)

        kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4

        return kp, desc

class ExtractORB(object):
    def __init__(self, num_kp):
        self.orb = cv2.ORB_create(nfeatures=num_kp)
    def run(self, img_path):
        img = cv2.imread(img_path, 0)
        kp = self.orb.detect(img, None)
        kp, desc = self.orb.compute(img, kp)

        kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in kp]) # N*4

        return kp, desc

def write_feature(pts, desc, filename):
  with h5py.File(filename, "w") as ifp:
      ifp.create_dataset('keypoints', pts.shape, dtype=np.float32)
      ifp.create_dataset('descriptors', desc.shape, dtype=np.float32)
      ifp["keypoints"][:] = pts
      ifp["descriptors"][:] = desc

if __name__ == "__main__":
    opt = parser.parse_args()
    if opt.suffix == 'sift-2000':
        detector = ExtractSIFT(opt.num_kp)
    elif opt.suffix == 'orb-2000':
        detector = ExtractORB(opt.num_kp)
    elif opt.suffix == 'super-2000':
        detector = ExtractSuper()
    else:
        raise RunTimeError("Unsupported detector")
    
    # get image lists
    search = os.path.join(opt.input_path, opt.img_glob)
    listing = glob.glob(search)

    for img_path in tqdm(listing):
      kp, desc = detector.run(img_path)
      save_path = img_path+'.'+opt.suffix+'.hdf5'
      write_feature(kp, desc, save_path)
