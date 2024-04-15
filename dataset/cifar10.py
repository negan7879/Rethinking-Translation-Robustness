import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json

import torch
from torch.utils.data import Dataset
from torch.utils import data
import cv2
import random
from einops import rearrange
from copy import deepcopy


class cifarDataset(Dataset):
    def __init__(self, train_mode="train", grid=5,
                 data_root="", canvas = "black", translate = 1, mode = 2):
        self.data_root = data_root
        self.mode = mode
        data_lists = os.listdir(self.data_root)
        data_lists.sort(key=lambda x: int(x.split('.')[0]))
        self.train_mode = train_mode
        self.canvas = canvas
        self.translate = translate
        assert self.canvas in ["black", "write", "mean"]
        assert self.translate in [1,3,5,7,9]
        if train_mode == "train":
            self.data_list = data_lists[:40000]
        elif train_mode == "val":
            self.data_list = data_lists[40000:]
        elif train_mode == "test":
            self.data_list = data_lists
        self.grid = grid

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        meta = h5py.File(os.path.join(self.data_root, self.data_list[index]), 'r')
        
        image = np.array(meta["image"]).astype(np.float32) / 255
        label = np.array(meta["label"]).item(0) + 1
        image[:, :, 0] = (image[:, :, 0] - mean[0]) / std[0]
        image[:, :, 1] = (image[:, :, 1] - mean[1]) / std[1]
        image[:, :, 2] = (image[:, :, 2] - mean[2]) / std[2]
        if self.train_mode == "train":
            if self.canvas == "black":
                canvas_img = np.zeros((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
            elif self.canvas == "write":
                canvas_img = np.ones((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
            elif self.canvas == "mean":
                canvas_img = np.full((32 * self.grid, 32 * self.grid, 3), (0.4914, 0.4822, 0.4465),
                                     dtype=np.float32)  
            canvas_label = np.zeros((32 * self.grid, 32 * self.grid), dtype=np.uint8)

            tmpmode = random.randint(0, 3)
            if tmpmode == 0:
                pos_x = 0
                pos_y = 0
            elif tmpmode == 1:
                pos_x = 0
                pos_y = (self.grid - 1) * 32
            elif tmpmode == 2:
                pos_x = (self.grid - 1) * 32
                pos_y = 0
            elif tmpmode == 3:
                pos_x = (self.grid - 1) * 32
                pos_y = (self.grid - 1) * 32
            assert pos_x < (32 * self.grid) and pos_y < (32 * self.grid)
            canvas_img[pos_x:pos_x + 32, pos_y: pos_y + 32] = image
            canvas_label[pos_x:pos_x + 32, pos_y: pos_y + 32] = label
            canvas_img = torch.from_numpy(canvas_img).permute([2, 0, 1])
            canvas_label = torch.from_numpy(canvas_label)
            return canvas_img, canvas_label
        elif self.train_mode == "val":
            if self.canvas == "black":
                canvas_img = np.zeros((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
            elif self.canvas == "write":
                canvas_img = np.ones((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
            elif self.canvas == "mean":
                canvas_img = np.full((32 * self.grid, 32 * self.grid, 3), (0.4914, 0.4822, 0.4465),
                                     dtype=np.float32)  
            canvas_label = np.zeros((32 * self.grid, 32 * self.grid), dtype=np.uint8)
            
            
            tmpmode = random.randint(0, 3)
            if tmpmode == 0:
                pos_x = 0
                pos_y = 0
            elif tmpmode == 1:
                pos_x = 0
                pos_y = (self.grid - 1) * 32
            elif tmpmode == 2:
                pos_x = (self.grid - 1) * 32
                pos_y = 0
            elif tmpmode == 3:
                pos_x = (self.grid - 1) * 32
                pos_y = (self.grid - 1) * 32
            assert pos_x < (32 * self.grid) and pos_y < (32 * self.grid)
            canvas_img[pos_x:pos_x + 32, pos_y: pos_y + 32] = image
            canvas_label[pos_x:pos_x + 32, pos_y: pos_y + 32] = label
            canvas_img = torch.from_numpy(canvas_img).permute([2, 0, 1])
            canvas_label = torch.from_numpy(canvas_label)
            return canvas_img, canvas_label
        elif self.train_mode == "test":
            mode = self.mode  
            canvas_imgs = []
            canvas_labels = []
            if mode == 1:
                for _ in range(5):
                    if self.canvas == "black":
                        canvas_img = np.zeros((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                    elif self.canvas == "write":
                        canvas_img = np.ones((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                    elif self.canvas == "mean":
                        canvas_img = np.full((32 * self.grid, 32 * self.grid, 3), (0.4914, 0.4822, 0.4465),
                                             dtype=np.float32)  
                    canvas_label = np.zeros((32 * self.grid, 32 * self.grid), dtype=np.uint8)
                    pos_x = random.randint(0, 32 * (self.grid - 1) - 1)
                    pos_y = random.randint(0, 32 * (self.grid - 1) - 1)
                    canvas_img[pos_x:pos_x + 32, pos_y: pos_y + 32] = image
                    canvas_label[pos_x:pos_x + 32, pos_y: pos_y + 32] = label
                    canvas_img_tensor = torch.from_numpy(canvas_img).permute([2, 0, 1])
                    canvas_label_tensor = torch.from_numpy(canvas_label)
                    canvas_imgs.append(canvas_img_tensor)
                    canvas_labels.append(canvas_label_tensor)
            elif mode == 2:
                for x in range(self.grid):
                    for y in range(self.grid):
                        if (x % 2 != 0) or (y % 2 != 0):
                            continue
                        pos_x = x * 32
                        pos_y = y * 32

                        pos_x += self.translate
                        if pos_x >= 32 * (self.grid - 1): continue
                        if self.canvas == "black":
                            canvas_img = np.zeros((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                        elif self.canvas == "write":
                            canvas_img = np.ones((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                        elif self.canvas == "mean":
                            canvas_img = np.full((32 * self.grid, 32 * self.grid, 3), (0.4914, 0.4822, 0.4465),
                                                 dtype=np.float32)  
                        canvas_label = np.zeros((32 * self.grid, 32 * self.grid), dtype=np.uint8)
                        canvas_img[pos_x:pos_x + 32, pos_y: pos_y + 32] = image
                        canvas_label[pos_x:pos_x + 32, pos_y: pos_y + 32] = label
                        canvas_img_tensor = torch.from_numpy(canvas_img).permute([2, 0, 1])
                        canvas_label_tensor = torch.from_numpy(canvas_label)
                        canvas_imgs.append(canvas_img_tensor)
                        canvas_labels.append(canvas_label_tensor)
            elif mode == 3: 
                for x in range(self.grid):
                    for y in range(self.grid):
                        if (x % 2 != 0) or (y % 2 != 0):
                            continue
                        pos_x = x * 32
                        pos_y = y * 32
                        if self.canvas == "black":
                            canvas_img = np.zeros((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                        elif self.canvas == "write":
                            canvas_img = np.ones((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                        elif self.canvas == "mean":
                            canvas_img = np.full((32 * self.grid, 32 * self.grid, 3), (0.4914, 0.4822, 0.4465),
                                                 dtype=np.float32)  
                        canvas_label = np.zeros((32 * self.grid, 32 * self.grid), dtype=np.uint8)
                        canvas_img[pos_x:pos_x + 32, pos_y: pos_y + 32] = image
                        canvas_label[pos_x:pos_x + 32, pos_y: pos_y + 32] = label
                        canvas_img_tensor = torch.from_numpy(canvas_img).permute([2, 0, 1])
                        canvas_label_tensor = torch.from_numpy(canvas_label)
                        canvas_imgs.append(canvas_img_tensor)
                        canvas_labels.append(canvas_label_tensor)
            elif mode == 4:  
                for x in range(self.grid):
                    for y in range(self.grid):
                        if (x % 2 == 0) and (y % 2 == 0):
                            continue
                        pos_x = x * 32
                        pos_y = y * 32
                        if self.canvas == "black":
                            canvas_img = np.zeros((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                        elif self.canvas == "write":
                            canvas_img = np.ones((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                        elif self.canvas == "mean":
                            canvas_img = np.full((32 * self.grid, 32 * self.grid, 3), (0.4914, 0.4822, 0.4465),
                                                 dtype=np.float32)  
                        canvas_label = np.zeros((32 * self.grid, 32 * self.grid), dtype=np.uint8)
                        canvas_img[pos_x:pos_x + 32, pos_y: pos_y + 32] = image
                        canvas_label[pos_x:pos_x + 32, pos_y: pos_y + 32] = label
                        canvas_img_tensor = torch.from_numpy(canvas_img).permute([2, 0, 1])
                        canvas_label_tensor = torch.from_numpy(canvas_label)
                        canvas_imgs.append(canvas_img_tensor)
                        canvas_labels.append(canvas_label_tensor)
            elif mode == 5:
                for _ in range(5):
                    if self.canvas == "black":
                        canvas_img = np.zeros((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                    elif self.canvas == "write":
                        canvas_img = np.ones((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                    elif self.canvas == "mean":
                        canvas_img = np.full((32 * self.grid, 32 * self.grid, 3), (0.4914, 0.4822, 0.4465),
                                             dtype=np.float32)  
                    canvas_label = np.zeros((32 * self.grid, 32 * self.grid), dtype=np.uint8)
                    pos_x = random.randint(0, (self.grid - 1))
                    pos_y = random.randint(int((self.grid - 1) /2) + 1, (self.grid - 1))
                    pos_x = pos_x * 32
                    pos_y = pos_y * 32
                    canvas_img[pos_x:pos_x + 32, pos_y: pos_y + 32] = image
                    canvas_label[pos_x:pos_x + 32, pos_y: pos_y + 32] = label
                    canvas_img_tensor = torch.from_numpy(canvas_img).permute([2, 0, 1])
                    canvas_label_tensor = torch.from_numpy(canvas_label)
                    canvas_imgs.append(canvas_img_tensor)
                    canvas_labels.append(canvas_label_tensor)
            elif mode == 6:
                for _ in range(5):
                    if self.canvas == "black":
                        canvas_img = np.zeros((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                    elif self.canvas == "write":
                        canvas_img = np.ones((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                    elif self.canvas == "mean":
                        canvas_img = np.full((32 * self.grid, 32 * self.grid, 3), (0.4914, 0.4822, 0.4465),
                                             dtype=np.float32)  
                    canvas_label = np.zeros((32 * self.grid, 32 * self.grid), dtype=np.uint8)
                    pos_x = random.randint(0, self.grid - 1)
                    pos_y = random.randint(0, int((self.grid - 1) / 2))
                    pos_x *= 32
                    pos_y *= 32
                    canvas_img[pos_x:pos_x + 32, pos_y: pos_y + 32] = image
                    canvas_label[pos_x:pos_x + 32, pos_y: pos_y + 32] = label
                    canvas_img_tensor = torch.from_numpy(canvas_img).permute([2, 0, 1])
                    canvas_label_tensor = torch.from_numpy(canvas_label)
                    canvas_imgs.append(canvas_img_tensor)
                    canvas_labels.append(canvas_label_tensor)
            elif mode == 7:
                for idx in range(4):
                    if self.canvas == "black":
                        canvas_img = np.zeros((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                    elif self.canvas == "write":
                        canvas_img = np.ones((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                    elif self.canvas == "mean":
                        canvas_img = np.full((32 * self.grid, 32 * self.grid, 3), (0.4914, 0.4822, 0.4465),
                                             dtype=np.float32)  
                    canvas_label = np.zeros((32 * self.grid, 32 * self.grid), dtype=np.uint8)
                    if idx == 0:
                        pos_x = 0
                        pos_y = 0
                    elif idx == 1:
                        pos_x = 0
                        pos_y = self.grid - 1
                    elif idx == 2:
                        pos_x = self.grid - 1
                        pos_y = 0
                    elif idx == 3:
                        pos_x = self.grid - 1
                        pos_y = self.grid - 1
                    pos_x *= 32
                    pos_y *= 32
                    canvas_img[pos_x:pos_x + 32, pos_y: pos_y + 32] = image
                    canvas_label[pos_x:pos_x + 32, pos_y: pos_y + 32] = label
                    canvas_img_tensor = torch.from_numpy(canvas_img).permute([2, 0, 1])
                    canvas_label_tensor = torch.from_numpy(canvas_label)
                    canvas_imgs.append(canvas_img_tensor)
                    canvas_labels.append(canvas_label_tensor)
            elif mode == 8:
                for idx in range(5):
                    if self.canvas == "black":
                        canvas_img = np.zeros((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                    elif self.canvas == "write":
                        canvas_img = np.ones((32 * self.grid, 32 * self.grid, 3), dtype=np.float32)  
                    elif self.canvas == "mean":
                        canvas_img = np.full((32 * self.grid, 32 * self.grid, 3), (0.4914, 0.4822, 0.4465),
                                             dtype=np.float32)  
                    canvas_label = np.zeros((32 * self.grid, 32 * self.grid), dtype=np.uint8)
                    pos_x = random.randint(0, (self.grid - 1))
                    if pos_x == 0 or pos_x == self.grid - 1 :
                        pos_y = random.randint(1, (self.grid - 2))
                    else:
                        pos_y = random.randint(0, (self.grid - 1))
                    pos_x *= 32
                    pos_y *= 32
                    canvas_img[pos_x:pos_x + 32, pos_y: pos_y + 32] = image
                    canvas_label[pos_x:pos_x + 32, pos_y: pos_y + 32] = label
                    canvas_img_tensor = torch.from_numpy(canvas_img).permute([2, 0, 1])
                    canvas_label_tensor = torch.from_numpy(canvas_label)
                    canvas_imgs.append(canvas_img_tensor)
                    canvas_labels.append(canvas_label_tensor)

            canvas_imgs = torch.stack(canvas_imgs, dim=0)
            canvas_labels = torch.stack(canvas_labels, dim=0)
            return canvas_imgs, canvas_labels

    
    
    
