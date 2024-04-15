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
from dataset.cifar10_center import cifarDataset_centor
from einops import rearrange

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = nn.NLLLoss(weight, reduction="mean")

    
    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)


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
grid_size = 5
data_root = ""
trainDataset = cifarDataset_centor(train_mode="train", grid=grid_size, data_root=data_root)
validDataset = cifarDataset_centor(train_mode="val", grid=grid_size, data_root=data_root)
trainloader = data.DataLoader(trainDataset,
                              batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
valloader = data.DataLoader(validDataset,
                            batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
num_classes = 11
model = FCN_tiny(3,num_classes).cuda()

params_to_optimize = [{"params": [p for p in model.parameters() if p.requires_grad]}]
lr = 0.01
opt = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-04, betas=(0.5, 0.999))
criterion = CrossEntropyLoss2d()
epochs = 100
miou_max = 0
exponent = 2
exp = 26
output_dir = "./output/exp{}/{}_grid_{}_exponent_{}_lr_{}".format(exp, model.__class__.__name__, grid_size, exponent,
                                                                  lr)
if os.path.exists(output_dir) is False:
    os.makedirs(output_dir)
for i in range(epochs):
    model.train()
    train_loss = 0.0
    for idx, (real_imgs, real_labels) in tqdm.tqdm(enumerate(trainloader)):
        opt.zero_grad()
        real_imgs = real_imgs.cuda()
        real_labels = real_labels.cuda().long()
        output = model(real_imgs)
        loss = criterion(output, real_labels)
        loss.backward()
        opt.step()
        train_loss = train_loss + loss.detach().item()
    train_loss = train_loss / len(trainloader)
    opt.param_groups[0]['lr'] = poly_lr(i, epochs, lr, exponent)
    with open(os.path.join(output_dir, "log.log"), "a") as fp:
        print("\nepoch = {}, loss = {}".format(i, train_loss), file=fp)

    if (i + 1) % 10 == 0:
        with torch.no_grad():
            model.eval()
            hist = np.zeros((num_classes, num_classes))
            for idx, (real_imgs, real_labels) in tqdm.tqdm(enumerate(valloader)):
                real_imgs = real_imgs.cuda()
                real_labels = real_labels.cuda().long()
                
                
                output = model(real_imgs)
                pred = torch.argmax(output, dim=1).long()
                hist += fast_hist(real_labels.data.cpu().numpy().flatten(), pred.data.cpu().numpy().flatten(),
                                  num_classes)
            mIoUs = per_class_iu(hist)
            
            miou = np.nanmean(mIoUs)
            if miou > miou_max:
                miou_max = miou
                torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pth"))
                with open(os.path.join(output_dir, "log.log"), "a") as fp:
                    print("get best MIOU = ", miou, file=fp)
            with open(os.path.join(output_dir, "log.log"), "a") as fp:
                print("epoch = {}, mIoUs = {}".format(i, miou), file=fp)
