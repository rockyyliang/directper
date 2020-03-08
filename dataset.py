import os
import json

import numpy as np
import cv2

import imgaug as ia
import imgaug.augmenters as iaa

import torch
from torch.utils.data import Dataset

from filehelp import make_file_name

# define augmenter
st = lambda aug: iaa.Sometimes(0.5, aug)
oc = lambda aug: iaa.Sometimes(0.3, aug)
rl = lambda aug: iaa.Sometimes(0.1,aug)
seq = iaa.Sequential([
    st(iaa.GaussianBlur((0, 1.5))),
    #rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0,0.05), per_channel=0.5)),
    #oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),
    oc(iaa.CoarseDropout((0.0, 0.05), size_percent=(0.08, 0.2),per_channel=0.5)),
    oc(iaa.Add((-0.25, 0.25), per_channel=0.5)),
    st(iaa.Multiply((0.8, 1.8), per_channel=0.2)),
    rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),
], random_order=True)

def hlc_to_onehot(hlc_int):
    hlc_int = int(hlc_int)
    output = np.zeros(5, dtype=int)
    if (hlc_int>=2) & (hlc_int<=6):
        #get position of one in return vector
        idx = hlc_int-2
        output[idx] = 1
        return output
    else:
        output[0] = 1
        #print('data hlc out of range! defaulting to follow lane', hlc_int)
        return output

class DPDataset(Dataset):
    def __init__(self, path, scales_dict, seq_len=5, separation=2, val_split=0.9, val=False, aug=False, idim=(88,200,3)):
        self.path = path
        self.recordings_list = sorted(os.listdir(self.path))
        self.recordings_num = len(self.recordings_list)

        #count number of datapoints and get integral list for recording search
        self.n_datapoints, self.recordings_dp_count = self._count_datapoints()
        self.recordings_integral = np.cumsum(self.recordings_dp_count)
        #print(self.recordings_dp_count)
        #print(self.recordings_integral)

        #validation things
        self.val = val
        self.val_split = val_split
        self.val_begin_idx = int(val_split*self.n_datapoints)

        #sequence
        self.seq_len = seq_len
        self.separation = separation

        self.idim = idim
        self.hlc_dim = 5

        self.aug = aug

        self.scale_dict = scales_dict


    def __len__(self):
        if self.val:
            usable_amount = self.n_datapoints - self.val_begin_idx
        else:
            usable_amount = self.val_begin_idx
        return usable_amount

    def __getitem__(self, idx):
        X, y = self.__generate_data(idx)
        return X, y

    def __generate_data(self, idx):
        '''
        y[0]: speed, distance to car, distance to pedestrian
        y[1]: curvature, heading error, crosstrack error
        '''
        if self.val:
            label_idx = idx + self.val_begin_idx
        else:
            label_idx = idx

        #find which recording label idx belongs to and index within that recording
        recording_idx = self._locate_recording(label_idx)
        label_idx_local = label_idx
        if recording_idx > 0:
            label_idx_local = label_idx - self.recordings_integral[recording_idx-1]

        #paths to take data from
        recording_path = self.recordings_list[recording_idx]
        rgb_path = os.path.join(self.path, recording_path, 'camera.rgb_0')
        label_path = os.path.join(self.path, recording_path, 'targets')

        #data file name
        jname = make_file_name(label_idx_local, prefix='target_', ext='.json')


        #get labels
        y_uncond = np.empty((3))
        y_cond = np.empty((3))
        with open(os.path.join(label_path, jname)) as j:
            data = json.load(j)
        y_uncond[0] = float(data['car_vx'])*self.scale_dict['car_vx']
        y_uncond[1] = float(data['dist_to_car'])*self.scale_dict['dist_to_car']
        y_uncond[2] = float(data['dist_to_walker'])*self.scale_dict['dist_to_walker']

        y_cond[0] = float(data['k'])*self.scale_dict['k']
        y_cond[1] = float(data['headingerror'])*self.scale_dict['headingerror']
        y_cond[2] = float(data['cte'])*self.scale_dict['cte']

        #loop to populate x array below
        X_p = np.empty((self.seq_len, *self.idim))
        X_s = np.empty((self.seq_len))
        X_c = np.empty((self.seq_len, self.hlc_dim))

        for inv_ts, d_idx in enumerate(range(label_idx_local, label_idx_local-(self.seq_len*self.separation), -self.separation)):
            #make sure d_idx doesn't go below 0
            d_idx = max(0, d_idx)
            #image name
            iname = make_file_name(d_idx, prefix='rgb_', ext='.jpeg')
            img = cv2.imread(os.path.join(rgb_path, iname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X_p[-1-inv_ts] = img/255.0

            jname = make_file_name(d_idx, prefix='target_', ext='.json')
            with open(os.path.join(label_path, jname)) as j:
                data = json.load(j)
            X_s[-1-inv_ts] = data['car_vx']

            #bandaid solution for bad hlc
            try:
                hlc = int(float(data['hlc']))
            except:
                print('hlc is bool in', os.path.join(label_path, jname))
                hlc = 2
            X_c[-1-inv_ts] = hlc_to_onehot(hlc)

        #augment images
        if self.aug:
            X_p = X_p.astype(np.float32)
            batch = ia.augmentables.batches.Batch(X_p)
            seq.augment_batch(batch)
            X_p = batch.images_aug

        #torch conversion
        X_p = np.moveaxis(X_p, 3, 1)
        X_list = [torch.as_tensor(x) for x in [X_p, X_s, X_c]]
        y_list = [torch.as_tensor(y) for y in [y_uncond, y_cond]]

        return X_list, y_list

    def _count_datapoints(self):
        '''get total number of datapoints'''
        total = 0
        recordings_dp_count = []
        for r in self.recordings_list:
            #get path where pics are stored
            rgb_path = os.path.join(self.path, r, 'camera.rgb_0')

            n = len(os.listdir(rgb_path))
            recordings_dp_count.append(n)

            total += n
        return total, recordings_dp_count

    def _locate_recording(self, idx):
        '''given total idx, find which recording it belongs to'''
        i = 0
        j = self.recordings_num-1
        while(i<=j):
            if self.recordings_integral[i] > idx:
                return i
            if self.recordings_integral[j] <= idx:
                return j + 1
            i += 1
            j -= 1

        #raise RuntimeError('two pointer search failed :( idx: {}, i: {}, j:{}'.format(idx, i, j))
        return i

if __name__=='__main__':
    import matplotlib.pyplot as plt

    dpath = 'Saved/'

    scale_dict =  {
        'car_vx':0.1,
        'dist_to_car':0.02,
        'dist_to_walker':0.02,
        'k':10,
        'headingerror':0.02,
        'cte':1
    }

    ds = DPDataset(dpath, scales_dict=scale_dict)

    X, y = ds[8]
    plt.imshow(np.moveaxis(X[0][0].numpy(),0,2))
    plt.show()
