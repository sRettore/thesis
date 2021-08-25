import os
import numpy as np
from torch.utils.data import Dataset
from torch import distributed, zeros_like, unique
import torchvision as tv
from PIL import Image
from .utils import Subset, filter_images, filter_images_with_weight
from .transform import MaskImageLabels

classes = [
    "unlabeled", 
    "ego vehicle",
    "rectification border",
    "out of roi", 
    "static",
    "dynamic",
    "ground",
    "road",
    "sidewalk",
    "parking",
    "rail track",
    "building",
    "wall",
    "fence",
    "guard rail",
    "bridge",
    "tunnel",
    "pole",
    "polegroup",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "caravan",
    "trailer",
    "train",
    "motorcycle",
    "bicycle",
    "license plate"
]

cities = [
    "aachen",
    "bochum",
    "bremen",
    "cologne",
    "darmstadt",
    "dusseldorf",
    "erfurt",
    "frankfurt",
    "hamhurg",
    "hanover",
    "jena",
    "krefeld",
    "lindau",
    "monchengladbach",
    "munster",
    "strasbourg",
    "stuttgart",
    "tubingen",
    "ulm"
    "weimar",
    "zurich"
]

class CityScapesDataset(Dataset):
    """ Simple segmentation dataset with both image datapoint and labels."""

    def __init__(self, root, image_set='train', is_aug=False, transform=None):
        super(CityScapesDataset, self).__init__()
        """
        Parameters
        ----------
        root : string
            Directory with all the images.
        image_set: string, optional
            Can be train, trainval or val, define the type of imageset to choose
        
        is_aug: bool, optional
            Choose if to use the image dataset that have been data augmented
        transform : callable, optional
            Optional transform to each image (note: use our shared transformation for both images).
        """
        
        self.root = os.path.join(root, "CityScapesDataset")
        self.transform = transform
        
        if not os.path.isdir(self.root):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(self.root, 'splits')
        
        if is_aug and (image_set == 'train' or image_set == 'trainval'):
            self.imagesDir = os.path.join(self.root,"images_aug")
            self.labelsDir = os.path.join(self.root,"labels_aug")
            
            if image_set == 'train':
                split_f = os.path.join(splits_dir, 'train_aug.txt')
            else:
                split_f = os.path.join(splits_dir, 'trainval_aug.txt')
        else:
            self.imagesDir = os.path.join(self.root,"images")
            self.labelsDir = os.path.join(self.root,"labels")
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
	
        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # filter images based on train, trainval, val
        self.fileList = [(x[0][0:], x[1][0:]) for x in file_names] # if os.path.isfile(os.path.join(self.imagesDir,x[0][0:])) and os.path.isfile(os.path.join(self.labelsDir,x[1][0:]))]
	
    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):            
        image = Image.open(os.path.join(self.imagesDir, self.fileList[idx][0])).convert('RGB')
        label = Image.open(os.path.join(self.labelsDir, self.fileList[idx][1]))
                        
        # apply the trasformations on both image and labels
        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label, idx
    
    def getFilename(self, idx):
        return os.path.basename(self.fileList[idx][0]).split('.')[0].rsplit('_', 1)[0]   
    
class CityScapesDatasetIncremental(Dataset):
    """ Segmentation dataset with both image datapoint and labels for incremental steps."""

    def __init__(self, 
                 root, 
                 train='train', 
                 is_aug=False, 
                 transform=None,
                 labels=None,
                 labels_old=None,
                 idxs_path=None,
                 masking=True,
                 overlap=True,
                 weight=0.,
                 where_to_sim='GPU_windows',
                 rank=0,
                 label_filter=False):
        super(CityScapesDatasetIncremental, self).__init__()
        """
        Parameters
        ----------
        root : string
            Directory with all the images.
        image_set: string, optional
            Can be train, trainval or val, define the type of imageset to choose
        is_aug: bool, optional
            Choose if to use the image dataset that have been data augmented
        transform : callable, optional
            Optional transform to each image (note: use our shared transformation for both images).
        """
        
        full_cts = CityScapesDataset(root, 'train' if train else 'val', is_aug=is_aug)
		
        self.labels = []
        self.labels_old = []

        self.where_to_sim = where_to_sim
        self.rank = rank

        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(l in labels_old for l in labels), "labels and labels_old must be disjoint sets"

            self.labels = [0] + labels
            self.labels_old = [0] + labels_old

            self.order = [0] + labels_old + labels
            
            """
            self.labels = labels
            self.labels_old = labels_old

            self.order = labels_old + labels
            """
            
            # take index of images with at least one class in labels and all classes in labels+labels_old+[0,255]
            if idxs_path is not None and os.path.exists(idxs_path):
                if weight > 0:
                    idxs = np.load(idxs_path)
                    self.weights = idxs['weights'].tolist()
                    idxs = idxs['idxs'].tolist()
                else:
                    idxs = np.load(idxs_path).tolist()
            else:
                print("Generating idxs_path for Cityscapes")	
                if weight > 0:
                    idxs, self.weights = filter_images_with_weight(full_cts, labels, weight, labels_old, overlap=overlap)

                    if idxs_path is not None and self.rank == 0:
                        np.save(idxs_path, idxs=idxs, weights=self.weights)
                else:
                    idxs = filter_images(full_cts, labels, labels_old, overlap=overlap)
                    
                    if idxs_path is not None and self.rank == 0:
                        np.save(idxs_path, idxs)

            if train or label_filter:   # label_filter force all unused labels to the 0 index and not the masking value
                masking_value = 0 #255
            else:
                masking_value = 255

            self.inverted_order = {label: self.order.index(label) for label in self.order}
            self.inverted_order[255] = masking_value
			
            if self.where_to_sim == 'GPU_windows' or self.where_to_sim == 'CPU_windows':
                reorder_transform = self.tmp_funct1
            else:
                reorder_transform = tv.transforms.Lambda(
                    lambda t: t.apply_(lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value))

            if masking:
                if self.where_to_sim == 'GPU_windows' or self.where_to_sim == 'CPU_windows':
                    target_transform = self.tmp_funct3
                else:
                    tmp_labels = self.labels + [255]
                    target_transform = tv.transforms.Lambda(
                        lambda t: t.apply_(lambda x: self.inverted_order[x] if x in tmp_labels else masking_value))
            else:
                target_transform = reorder_transform

            # make the subset of the dataset
            self.dataset = Subset(full_cts, idxs, transform, target_transform)
        else:
            self.dataset = full_cts

    def labelsName(self):
        return [classes[label] for label in self.order]
        
    def tmp_funct1(self, x):
        tmp = zeros_like(x)
        for value in unique(x):
            if value in self.inverted_order:
                new_value = self.inverted_order[value.item()]
            else:
                new_value = self.inverted_order[255]  # i.e. masking value
            tmp[x == value] = new_value
        return tmp

    def tmp_funct3(self, x):
        tmp = zeros_like(x)
        for value in unique(x):
            if value in self.labels + [255]:
                new_value = self.inverted_order[value.item()]
            else:
                new_value = self.inverted_order[255]  # i.e. masking value
            tmp[x == value] = new_value
        return tmp

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)