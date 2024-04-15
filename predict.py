import warnings

warnings.filterwarnings("ignore")
import torch.utils.data
import torch
import numpy as np
import shutil
import os
import torch.nn.functional as F
from torch import nn
from torch.utils import data
import tqdm
from models.model import FCN, ResNet18, FCN_tiny 
from dataset.cifar10 import cifarDataset
from einops import rearrange
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def fast_hist(a, b, n):  

    k = (a >= 0) & (a < n)  
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n,
                                                                              n)  


def per_class_iu(hist):  

    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(3407)
canvas="black"
grid_size = 5
translate = 5
mode = 1

data_root = "/media/data/zhangzherui/code/myIDEA/data/cifar-10-batches-py/hdf5_test"


testDataset = cifarDataset(train_mode="test", grid=grid_size, data_root=data_root,canvas=canvas, translate=translate,mode=mode)
testloader = data.DataLoader(testDataset,
                             batch_size=1, shuffle=True, pin_memory=True, num_workers=4)
num_classes = 11


model = FCN_tiny(3, num_classes).cuda()






output_dir = "/media/data/zhangzherui/code/myIDEA/output/exp61/FCN_tiny_grid_5_exponent_2_lr_0.001_aug_10_canvas_black"

checkpoint = torch.load(os.path.join(output_dir, "model_best.pth"))
model.load_state_dict(checkpoint)
print("load model success !!!")


def show_label(data, index):
    plt.imshow(data[index])
    plt.show()


def show_img(data, index):
    plt.imshow(data[index])
    plt.show()

difference = []
with torch.no_grad():
    model.eval()

    for idx, (real_imgs, real_labels) in tqdm.tqdm(enumerate(testloader)):
        
        real_imgs = real_imgs.cuda()
        real_labels = real_labels.cuda().long()
        real_imgs = rearrange(real_imgs, 'b s c h w -> (b s) c h w')
        real_labels = rearrange(real_labels, 'b s h w -> (b s) h w')
        real_imgs_NP = real_imgs.data.cpu().numpy()
        real_labels_NP = real_labels.data.cpu().numpy()
        output = model(real_imgs)
        pred = torch.argmax(output, dim=1).long()
        pred_NP = pred.data.cpu().numpy()
        real_labels_NP[real_labels_NP > 0] = 1
        pred_NP[pred_NP > 0] = 1
        nums = []
        for index in range(len(real_labels_NP)):
            hist = fast_hist(real_labels_NP[index].flatten(), pred_NP[index].flatten(), 2)
            mIoUs = per_class_iu(hist)
            miou = np.nanmean(mIoUs)
            nums.append(miou)
        if mode == 2:
            difference.append(nums[0] - nums[1])
        elif mode == 1:
            difference.extend(nums)

    print("res ", np.mean(difference))
    
