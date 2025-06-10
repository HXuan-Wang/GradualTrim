import os
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class Data:
    def __init__(self, args, is_evaluate=False):
        pin_memory = False
        if args.gpu is not None:
            pin_memory = True

        scale_size = 224

        traindir = os.path.join(args.data_dir, 'train')
        valdir = os.path.join(args.data_dir, 'eval')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if not is_evaluate:
            self.trainset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(scale_size),
                    transforms.ToTensor(),
                    normalize,
                ]))

            self.loader_train = DataLoader(
                self.trainset,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=pin_memory)

        self.testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]))

        self.loader_test = DataLoader(
            self.testset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class FewShotImageFolder(torch.utils.data.Dataset):
    # set default seed=None, check the randomness
    def __init__(self, root, transform=None, N=1000, K=-1, few_samples=-1, seed=None):
        super(FewShotImageFolder, self).__init__()
        self.root = os.path.abspath(os.path.expanduser(root))
        self._transform = transform
        # load and parse from a txt file
        self.N = N
        self.K = K
        self.few_samples = few_samples
        self.seed = seed
        self.samples = self._parse_and_sample()

    def samples_to_file(self, save_path):
        with open(save_path, "w") as f:
            for (path, label) in self.samples:
                f.writelines("{}, {}\n".format(path.replace(self.root, "."), label))
        print("Writing train samples into {}".format(os.path.abspath(save_path)))

    def __parse(self):
        file_path = os.path.join(self.root, "train.txt")
        full_data = {}
        with open(file_path, "r") as f:
            raw_data = f.readlines()
        for rd in raw_data:
            img_path, target = rd.replace("\n", "").split()
            assert target.isalnum()
            if target not in full_data.keys():
                full_data[target] = []
            full_data[target].append(img_path)
        return full_data

    def _parse_and_sample(self):
        N, K, seed = self.N, self.K, self.seed
        assert 1 <= N <= 1000, r"N with maximum num 1000"
        assert K <= 500, r"If you want to use the whole dataset, set K=-1"
        # txt default path: self.root + "/train.txt"
        full_data = self.__parse()
        all = 0
        for v in full_data.values():
            all += len(v)
        print("Full dataset has {} classes and {} images.".format(len(full_data), all))
        print("Using seed={} to sample images.".format(seed))
        sampled_data = []

        np.random.seed(seed)
        # sample classes
        if self.few_samples > 0:
            for i in range(self.few_samples):
                while True:
                    sampled_cls = np.random.choice(list(full_data.keys()), 1, replace=False)
                    cls = sampled_cls[0]
                    sampled_img = np.random.choice(full_data[cls], 1, replace=False)[0]
                    curr_sample = (os.path.join(self.root, "train", sampled_img), cls)
                    if curr_sample not in sampled_data:
                        sampled_data.append(curr_sample)
                        break
            print("Final samples: {}".format(len(sampled_data)))
        else:
            sampled_cls = np.random.choice(list(full_data.keys()), N, replace=False)
            sampled_cls.sort()
            for cls in sampled_cls:
                if K == -1:
                    # use all data
                    sampled_imgs = full_data[cls]
                else:
                    # sample images of every class
                    sampled_imgs = np.random.choice(full_data[cls], K, replace=False)
                sampled_data += [(os.path.join(self.root, "train", i), cls) for i in sorted(sampled_imgs)]

        self.idx_to_class = {}
        self.class_to_idx = {}
        for k, v in full_data.items():
            idx = k
            cls = v[0].split("/")[0]
            self.class_to_idx[cls] = idx
            self.idx_to_class[idx] = cls
        self.classes = list(self.idx_to_class.values())
        self._full_data = full_data
        return sampled_data

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = pil_loader(path)
        if self._transform is not None:
            img = self._transform(img)
        return img, int(label)

    def __len__(self):
        return len(self.samples)

    def __repr__(self) -> str:
        return super().__repr__()


def imagenet_fewshot(img_num=1000, batch_size=64, seed=2021, data_dir=None, train=True):
    if img_num < 1000:
        few_samples = img_num
        N = 1000
        K = -1
    else:
        few_samples = -1
        N = 1000
        K = img_num // N

    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        shuffle = True
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        shuffle = False

    dataset = FewShotImageFolder(
        data_dir,
        transform,
        N=N, K=K, few_samples=few_samples, seed=seed)

    drop_last = False
    # if train and len(dataset) >= batch_size:
    #     drop_last = True

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=8, pin_memory=False, drop_last=drop_last
    )

    return loader

def eval_imagenet_fewshot(img_num=1000, batch_size=64, seed=2021, data_dir=None, train=True):
    if img_num < 1000:
        few_samples = img_num
        N = 1000
        K = -1
    else:
        few_samples = -1
        N = 1000
        K = img_num // N

    if train:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        shuffle = False
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        shuffle = False

    dataset = FewShotImageFolder(
        data_dir,
        transform,
        N=N, K=K, few_samples=few_samples, seed=seed)

    drop_last = False
    if train and len(dataset) >= batch_size:
        drop_last = True

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=4, pin_memory=False, drop_last=False
    )
    return loader