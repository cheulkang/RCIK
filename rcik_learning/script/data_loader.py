import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self, dirpath, dof, isRobotBody=False):
        self.len_wsi = 2000
        self.num_data_each_wsi = 500
        self.num_data_each_coll_wsi = 250
        self.num_data_each_free_wsi = self.num_data_each_wsi - self.num_data_each_coll_wsi
        self.dirpath = dirpath
        self.dof = dof
        self.isRobotBody = isRobotBody
        self.x_wsi = torch.from_numpy(self._load_workSpaceInfo())
        x, y = self._load_stateInfo()
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.x, self.y = self.x.type(torch.FloatTensor), self.y.type(torch.FloatTensor)
        self.x_wsi = self.x_wsi.type(torch.FloatTensor)
        self.len = self.x.shape[0]
        self.target = []
        for i in range(self.len_wsi):
            self.target.extend([i]*self.num_data_each_wsi)

    def _load_stateInfo(self):
        input = []
        output = []
        for i in range(0, self.len_wsi):
            rp = self.dirpath + 'gt/sdf_coll_' + str(i) + '.npy'
            data = np.load(rp)[:self.num_data_each_coll_wsi]
            input.extend(data[:, :self.dof])
            output.extend(data[:, self.dof:])

            rp = self.dirpath + 'gt/sdf_free_' + str(i) + '.npy'
            data = np.load(rp)[:self.num_data_each_free_wsi]
            input.extend(data[:, :self.dof])
            output.extend(data[:, self.dof:])

        return np.asarray(input), np.asarray(output)

    def _load_workSpaceInfo(self):
        wsi = []
        if(self.isRobotBody):
            rp = self.dirpath + 'self_distance_field.npy'
            self_dp = np.load(rp)
            for i in range(0, self.len_wsi):
                rp = self.dirpath + '../fetch_arm/occ_05/occ_' + str(i) + '.npy'
                wsi.append(np.load(rp)+ self_dp)
        else:
            for i in range(0, self.len_wsi):
                rp = self.dirpath + '../fetch_arm/occ_05/occ_' + str(i) + '.npy'
                wsi.append(np.load(rp))

        return np.asarray(wsi)

    def __getitem__(self, index):
        a = self.x[index]
        # a = a.view(1, a.size(0))
        b = self.x_wsi[int(index / self.num_data_each_wsi)]
        b = b.view(1, b.size(0), b.size(1), b.size(2))  # 216
        return a, b, self.y[index]

    def __len__(self):
        return self.len

class ValidationDataset(Dataset):
    def __init__(self, dirpath, dof, num_scene, start_scene, num_pose=10, isRobotBody=False):
        self.len_wsi = num_scene
        self.start = start_scene
        self.num_pose = num_pose
        self.num_state = 50
        self.num_data_each_wsi = self.num_pose * self.num_state
        self.dirpath = dirpath
        self.dof = dof
        self.isRobotBody = isRobotBody

        self.x_wsi = torch.from_numpy(self._load_workSpaceInfo())
        x = self._load_stateInfo()
        self.x = torch.from_numpy(x)
        self.x = self.x.type(torch.FloatTensor)
        self.x_wsi = self.x_wsi.type(torch.FloatTensor)
        self.len = self.x.shape[0]

    def _load_stateInfo(self):
        input = []
        for i in range(self.len_wsi):
            for j in range(self.num_pose):
                rp = self.dirpath + 'gt_val/sdf_data_' + str(self.start + i) + '_' + str(j) + '.npy'
                data = np.load(rp)[:self.num_state]
                input.extend(data)

        return np.asarray(input)

    def _load_workSpaceInfo(self):
        wsi = []
        if (self.isRobotBody):
            rp = self.dirpath + 'self_distance_field.npy'
            self_dp = np.load(rp)
            for i in range(0, self.len_wsi):
                rp = self.dirpath + '../fetch_arm/occ_05/occ_' + str(self.start + i) + '.npy'
                wsi.append(np.load(rp) + self_dp)
        else:
            for i in range(0, self.len_wsi):
                rp = self.dirpath + '../fetch_arm/occ_05/occ_' + str(self.start + i) + '.npy'
                wsi.append(np.load(rp))
        return np.asarray(wsi)

    def __getitem__(self, i):
        a = self.x[i][:self.dof]
        y = self.x[i][self.dof:]
        b = self.x_wsi[int(i / self.num_data_each_wsi)]
        b = b.view(1, b.size(0), b.size(1), b.size(2)) # 216
        return a, b, y

    def __len__(self):
        return self.len
