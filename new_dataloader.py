import torch
from torch.utils.data import Dataset
import glob
import os
import pydicom
import nibabel as nib
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_transform = target_transform
        self.target = target_transform
        self.patients = glob.glob(f'{image_dir}/MTR*')
        self.patients.sort()
        self.num_to_patients = {i:n[-3:] for i,n in enumerate(self.patients)}
        self.slice_to_fid = {num: 2 * num + 1 for num in range(161)}

    def __len__(self):
        return 2*160

    def __getitem__(self, idx):
        # get patient and slice corresponding to this index
        num = idx // 160
        slice_idx = idx % 160
        patient = self.num_to_patients[num]
        fid_slice = self.slice_to_fid[slice_idx]

        # access slice
        image_folder = os.path.join(self.image_dir, f'MTR_{patient:03}')
        image_path = os.path.join(image_folder, f'I{fid_slice:03}.dcm')
        dicom_img = pydicom.dcmread(image_path)
        image = dicom_img.pixel_array

        # access label
        label_path = os.path.join(self.label_dir, f'MTR_{patient:03}.nii')
        loaded_volume = nib.load(label_path)
        label_volume = loaded_volume.get_fdata()
        label = label_volume[:, :, slice_idx]
        label = np.rot90(label, k =-1)
        label = np.fliplr(label)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label






