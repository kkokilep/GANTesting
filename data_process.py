import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def load_data(opt):
    splits = ['train','test']
    drop_last_batch = {'train': True,  'test': False}
    shuffle = {'train': True, 'test': True}
    b_size = {'train':opt.batchsize,'test': 1}
    transform = transforms.Compose([transforms.Resize(opt.im_size),
                                    transforms.CenterCrop(opt.im_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])
    dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), transform) for x in splits}

    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=b_size[x],
                                                 shuffle=shuffle[x],
                                                 num_workers=int(opt.workers),
                                                 drop_last=drop_last_batch[x],
                                                 worker_init_fn=(None if opt.manualseed == -1
                                                                 else lambda x: np.random.seed(opt.manualseed)))
                  for x in splits}
    return dataloader
