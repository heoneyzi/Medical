import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class Custom2DBraTSDataset(Dataset):
    def __init__(self, data_dir, modality):
        self.data_dir = data_dir
        self.modality = modality
        self.patient_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        # Initialize lists to store slices
        self.images = []
        self.labels = []

        # Iterate through patients and load slices
        for patient_id in self.patient_ids:
            patient_path = os.path.join(self.data_dir, patient_id)

            # Load image and label volumes
            image = nib.load(os.path.join(patient_path, f'{patient_id}_{self.modality}.nii.gz')).get_fdata()
            label = nib.load(os.path.join(patient_path, f'{patient_id}_seg.nii.gz')).get_fdata()

            # Append all slices to the list
            for slice_idx in range(image.shape[2] // 2 - 5, image.shape[2] // 2 + 5):
                image_slice = image[:, :, slice_idx]
                label_slice = label[:, :, slice_idx]

                # Convert to torch tensor and add channel dimension for image
                image_tensor = torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

                # Resize image and label to 256x256
                image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
                label_slice = torch.tensor(label_slice, dtype=torch.float32).unsqueeze(0)  # Add channel dimension for resizing
                label_tensor = F.interpolate(label_slice.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)

                label_tensor[label_tensor == 4] = 3

                self.images.append(image_tensor)
                self.labels.append(label_tensor)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.images)

    def preprocess_mask_labels(self, mask: np.ndarray):
        mask_WT = np.zeros(mask.shape)
        mask_WT[mask == 2] = 1

        mask_TC = np.zeros(mask.shape)
        mask_TC[mask == 1] = 1

        mask_ET = np.zeros(mask.shape)
        mask_ET[mask == 3] = 1

        mask_BG = np.zeros(mask.shape)
        mask_BG[mask == 0] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET, mask_BG])
        return mask
    
    
data_dir = '../data/BraTS_2018_Train' # put the directory of data
modalities = ['t1', 't1ce', 't2', 'flair']

t1_dataset = Custom2DBraTSDataset(data_dir=data_dir, modality='t1')
t2_dataset = Custom2DBraTSDataset(data_dir=data_dir, modality='t2')
t1ce_dataset = Custom2DBraTSDataset(data_dir=data_dir, modality='t1ce')
flair_dataset = Custom2DBraTSDataset(data_dir=data_dir, modality='flair')

datasets = t1_dataset
datasets = datasets + t2_dataset + t1ce_dataset + flair_dataset

len(t1_dataset)
len(datasets)

# 데이터셋이 이미 생성된 리스트를 파일로 저장
def save_dataset(images, labels, filepath):
    data = {'images': images, 'labels': labels}
    torch.save(data, filepath)

# 파일에서 데이터셋을 불러오는 클래스
class LoadDatasetFromDisk(Dataset):
    def __init__(self, filepath):
        data = torch.load(filepath)
        self.images = data['images']
        self.labels = data['labels']

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)

# 사용 예
# 데이터셋 저장
save_dataset(t1_dataset.images, t1_dataset.labels, '../data/BraTS_2018_Training_dataset')

