import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted
from PIL import Image
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# 对数据集图像进行处理
transform = T.Compose([
    T.RandomCrop(128),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

# 使用 albumentations 库对图像进行处理
transform_A = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.RandomRotate90(),
    A.augmentations.transforms.ChannelShuffle(0.6),
    ToTensorV2()
])
transform_A_valid = A.Compose([
    A.CenterCrop(width=256, height=256),
    ToTensorV2()
])
DIV2K_path = "/data/whq/data/DIV2K"

batchsize = 16

# dataset
class DIV2K_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            self.files = natsorted(
                sorted(glob.glob(DIV2K_path+"/DIV2K_train_HR"+"/*."+"png")))
        else:
            self.files = natsorted(
                sorted(glob.glob(DIV2K_path+"/DIV2K_valid_HR"+"/*."+"png")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        trans_img = self.transform(image=img)
        item= trans_img['image']
        item=item/255.0
        return item

    def __len__(self):
        return len(self.files)


# dataloader
DIV2K_train_cover_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=batchsize,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)

DIV2K_train_secret_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=batchsize,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)

DIV2K_val_cover_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=batchsize,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)

DIV2K_val_secret_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=batchsize,
    shuffle=False,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)
