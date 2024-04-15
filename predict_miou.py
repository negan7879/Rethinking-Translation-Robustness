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
from models.model import FCN, ResNet18, FCN_tiny,ResNet34_class 
from dataset.cifar10 import cifarDataset
from einops import rearrange
import matplotlib.pyplot as plt
import argparse



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

def show_label(data, index):
    plt.imshow(data[index])
    plt.show()

def show_img(data, index):
    plt.imshow(data[index])
    plt.show()


def func(canvas, grid_size, translate, mode, data_root, output_dir,model_):
    assert model_ in ["fcn", "resnet18", "resnet34"]
    num_classes = 11
    if model_ == "fcn":
        model = FCN_tiny(3, num_classes).cuda()
        bs = 25
    elif model_ == "resnet18":
        model = ResNet18(num_classes).cuda()
        bs = 20
    elif model_ == "resnet34":
        model = ResNet34_class(n_class=num_classes).cuda()
    testDataset = cifarDataset(train_mode="test", grid=grid_size, data_root=data_root, canvas=canvas,
                               translate=translate, mode=mode)
    testloader = data.DataLoader(testDataset,
                                 batch_size=bs, shuffle=False, pin_memory=True, num_workers=8)
    checkpoint = torch.load(os.path.join(output_dir, "model_best.pth"))
    model.load_state_dict(checkpoint)
    print("load model success !!!")
    with torch.no_grad():
        model.eval()
        hist = np.zeros((11, 11))
        for idx, (real_imgs, real_labels) in tqdm.tqdm(enumerate(testloader)):
            
            real_imgs = real_imgs.cuda()
            real_labels = real_labels.cuda().long()
            real_imgs = rearrange(real_imgs, 'b s c h w -> (b s) c h w')
            real_labels = rearrange(real_labels, 'b s h w -> (b s) h w')
            real_imgs_NP = real_imgs.data.cpu().numpy()
            real_labels_NP = real_labels.data.cpu().numpy()
            output = model(real_imgs)
            pred = torch.argmax(output, dim=1).long()
            hist += fast_hist(real_labels.data.cpu().numpy().flatten(), pred.data.cpu().numpy().flatten(),
                              11)

        mIoUs = per_class_iu(hist)
        miou = np.nanmean(mIoUs)
        with open(os.path.join(output_dir, "result.log"), "a") as fp:
            print("mode = {} translate = {}  miou = {}".format(mode, translate, miou), file=fp)







