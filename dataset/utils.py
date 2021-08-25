import torch
import numpy as np

def group_images(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if all(x in labels_cum for x in cls):
            for x in cls:
                if x in labels:
                    idxs[x].append(i)
    return idxs

def filter_images_with_weight(dataset, labels, weight=1., labels_old=None, overlap=True):
    # Filter images without any label in LABELS (using labels not reordered)
    idxs = []
    counts = []
    weights = []
    
    if 0 in labels:
        labels.remove(0)

    if labels_old is None:
        labels_old = []
    labels_cum = labels + labels_old + [0, 255]
    order = [0] + labels 
    num_classes = len (order)
    
    classCounters = np.zeros(num_classes)
    inverted_order = {label: order.index(label) for label in order}
    
    if overlap:
        fil = lambda c: any(x in labels for x in cls)
    else:  # disjoint and no_mask (ICCVW2019) datasets are the same, only label space changes
        fil = lambda c: any(x in labels for x in cls) and all(x in labels_cum for x in c)

    for i in range(len(dataset)):
        cls, count = np.unique(np.array(dataset[i][1]), return_counts=True)
        cls = np.unique(np.array(dataset[i][1]))
        if fil(cls):
            idxs.append(i)
            counted0 = False
            #counts.append(count)
            #print(count)
            sample_count = np.zeros(num_classes)
            for j in range(len(cls)):
                if cls[j] == 0:
                    counted0 = True
                if cls[j] in order:
                    classCounters[inverted_order[cls[j]]] += 1
                    sample_count[inverted_order[cls[j]]] = 1# += count[j]
                else:
                    if not counted0:
                        classCounters[0] += 1
                        counted0 = True
                    if len(sample_count) != 0:
                        sample_count[0] = 1 #count[j]
                    else:
                        sample_count[inverted_order[cls[j]]] = 1 #+= count[j]
            counts.append(sample_count)
        if i % 1000 == 0:
            print(f"\t{i}/{len(dataset)} ...")
    print("classcaunters: ", classCounters)
    classCounters = np.power(len(counts) / classCounters, weight)
    print("classcaunters: ", classCounters)
    
    for sample_count in counts:
        #print(np.sum(classCounters*sample_count)/np.sum(sample_count))
        weights.append(np.sum(classCounters*sample_count)/np.sum(sample_count))
        #print(np.sum(classCounters*sample_count)/np.sum(sample_count))
        #weights.append(np.sum(classCounters*sample_count)/np.sum(sample_count))
    print("min max avg", np.min(weights)," , ", np.max(weights)," , ", np.mean(weights))
    print("counts len", len(counts))
    print("idxs   len", len(idxs))
    print("weightslen", len(weights))
    return np.asarray(idxs, dtype=int), np.asarray(weights)

def filter_images(dataset, labels, labels_old=None, overlap=True):
    # Filter images without any label in LABELS (using labels not reordered)
    idxs = []

    if 0 in labels:
        labels.remove(0)

    if labels_old is None:
        labels_old = []
    labels_cum = labels + labels_old + [0, 255]

    if overlap:
        fil = lambda c: any(x in labels for x in cls)
    else:  # disjoint and no_mask (ICCVW2019) datasets are the same, only label space changes
        fil = lambda c: any(x in labels for x in cls) and all(x in labels_cum for x in c)

	
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if fil(cls):
            idxs.append(i)
        if i % 1000 == 0:
            print(f"\t{i}/{len(dataset)} ...")

    return np.asarray(idxs, dtype=int)


class Subset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (callable): way to transform the images and the targets
        target_transform(callable): way to transform the target labels
    """

    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        sample, target, index = self.dataset[self.indices[idx]]

        if self.transform is not None:
            sample, target = self.transform(sample, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def __len__(self):
        return len(self.indices)


class MaskLabels:
    """
    Use this class to mask labels that you don't want in your dataset.
    Arguments:
    labels_to_keep (list): The list of labels to keep in the target images
    mask_value (int): The value to replace ignored values (def: 0)
    """
    def __init__(self, labels_to_keep, mask_value=0):
        self.labels = labels_to_keep
        self.value = torch.tensor(mask_value, dtype=torch.uint8)

    def __call__(self, sample):
        # sample must be a tensor
        assert isinstance(sample, torch.Tensor), "Sample must be a tensor"

        sample.apply_(lambda t: t.apply_(lambda x: x if x in self.labels else self.value))

        return sample
