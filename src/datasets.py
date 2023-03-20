
import json
import random
import torch
import torchvision.io as io
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class Adobe5kPairedDataset(Dataset):
    def __init__(self, db_path, root, training=True):
        with open(db_path, 'rt') as f:
            self.db = json.load(f)
        self.root = root
        self.training = training

    def __len__(self):
        return len(self.db["label"])

    def transform(self, image, label):
        # normalize
        image = image.float() / 127.5 - 1.0
        label = label.float() / 127.5 - 1.0

        if self.training:
            H, W = image.size(1), image.size(2)
            # random crop and resize
            h = random.randint(H // 2, H)
            w = random.randint(W // 2, W)
            t = random.randint(0, H - h)
            l = random.randint(0, W - w)
            image = TF.crop(image, t, l, h, w)
            label = TF.crop(label, t, l, h, w)
            image = TF.resize(image, (256, 256))
            label = TF.resize(label, (256, 256))

            # random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
            
            # random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)

        H, W = image.size(1), image.size(2)
        if (H % 4) or (W % 4):
            nH = (H // 4) * 4
            nW = (W // 4) * 4
            image = TF.resize(image, (nH, nW))
            label = TF.resize(label, (nH, nW))
                        
        return image, label

    def __getitem__(self, idx):
        image_name = self.db["image"][idx]
        label_name = self.db["label"][idx]
        image_path = f"{self.root}/{image_name}"
        label_path = f"{self.root}/{label_name}"    
        image = io.read_image(image_path)
        label = io.read_image(label_path)

        image, label = self.transform(image, label)

        return image, label

import glob
class LoLPairedDataset(Dataset):
    def __init__(self, db_path, root, training=True):
        with open(db_path, 'rt') as f:
            self.db = json.load(f)
        self.root = root
        self.training = training
        
    def __len__(self):
        return len(self.db["label"])

    def transform(self, image, label):
        # normalize
        image = image.float() / 127.5 - 1.0
        label = label.float() / 127.5 - 1.0

        if self.training:
            H, W = image.size(1), image.size(2)
            # random crop and resize
            h = random.randint(H // 2, H)
            w = random.randint(W // 2, W)
            t = random.randint(0, H - h)
            l = random.randint(0, W - w)
            image = TF.crop(image, t, l, h, w)
            label = TF.crop(label, t, l, h, w)
            image = TF.resize(image, (256, 256))
            label = TF.resize(label, (256, 256))

            # random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
            
            # random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)

            # if random.random() > 0.5:
            #    noise_level = random.uniform(0.0, 5.0)
            #    noise = torch.normal(0.0, 1.0, size=image.size())
            #    noise = noise / torch.norm(noise, p=2)
            #    image = image + noise_level * noise

        H, W = image.size(1), image.size(2)
        if (H % 4) or (W % 4):
            nH = (H // 4) * 4
            nW = (W // 4) * 4
            image = TF.resize(image, (nH, nW))
            label = TF.resize(label, (nH, nW))
                        
        return image, label
    
    def __getitem__(self, idx):
        image_name = self.db["image"][idx]
        label_name = self.db["label"][idx]
        image_path = f"{self.root}/{image_name}"
        label_path = f"{self.root}/{label_name}"     
        image = io.read_image(image_path)
        label = io.read_image(label_path)

        image, label = self.transform(image, label)

        return image, label

class VELoLPairedDataset(Dataset):
    def __init__(self, db_path, root, training=True):
        with open(db_path, 'rt') as f:
            self.db = json.load(f)
        self.root = root
        self.training = training

    def __len__(self):
        return len(self.db["label"])

    def transform(self, image, label):
        # normalize
        image = image.float() / 127.5 - 1.0
        label = label.float() / 127.5 - 1.0

        if self.training:
            H, W = image.size(1), image.size(2)
            # random crop and resize
            h = random.randint(H // 2, H)
            w = random.randint(W // 2, W)
            t = random.randint(0, H - h)
            l = random.randint(0, W - w)
            image = TF.crop(image, t, l, h, w)
            label = TF.crop(label, t, l, h, w)
            image = TF.resize(image, (256, 256))
            label = TF.resize(label, (256, 256))

            # random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
            
            # random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)

            # if random.random() > 0.5:
            #    noise_level = random.uniform(0.0, 5.0)
            #    noise = torch.normal(0.0, 1.0, size=image.size())
            #    noise = noise / torch.norm(noise, p=2)
            #    image = image + noise_level * noise

        H, W = image.size(1), image.size(2)
        if (H % 4) or (W % 4):
            nH = (H // 4) * 4
            nW = (W // 4) * 4
            image = TF.resize(image, (nH, nW))
            label = TF.resize(label, (nH, nW))
                        
        return image, label

    def __getitem__(self, idx):
        image_name = self.db["image"][idx]
        label_name = self.db["label"][idx]
        image_path = f"{self.root}/{image_name}"
        label_path = f"{self.root}/{label_name}"     
        image = io.read_image(image_path)
        label = io.read_image(label_path)

        image, label = self.transform(image, label)

        return image, label


class HDRplusDataset(Dataset):
    def __init__(self, db_path, root, training=True):
        with open(db_path, 'rt') as f:
            self.db = json.load(f)
        self.root = root
        self.training = training

    def __len__(self):
        return len(self.db["label"])

    def transform(self, image, label):
        # normalize
        image = image.float() / 127.5 - 1.0
        label = label.float() / 127.5 - 1.0

        if self.training:
            H, W = image.size(1), image.size(2)
            # random crop and resize
            h = random.randint(H // 2, H)
            w = random.randint(W // 2, W)
            t = random.randint(0, H - h)
            l = random.randint(0, W - w)
            image = TF.crop(image, t, l, h, w)
            label = TF.crop(label, t, l, h, w)
            image = TF.resize(image, (256, 256))
            label = TF.resize(label, (256, 256))

            # random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
            
            # random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)

        H, W = image.size(1), image.size(2)
        if (H % 4) or (W % 4):
            nH = (H // 4) * 4
            nW = (W // 4) * 4
            image = TF.resize(image, (nH, nW))
            label = TF.resize(label, (nH, nW))
                        
        return image, label

    def __getitem__(self, idx):
        image_name = self.db["image"][idx]
        label_name = self.db["label"][idx]
        image_path = f"{self.root}/{image_name}"
        label_path = f"{self.root}/{label_name}"    
        image = io.read_image(image_path)
        label = io.read_image(label_path)

        image, label = self.transform(image, label)

        return image, label
