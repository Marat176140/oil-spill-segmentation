import os
import cv2
import numpy as np
import kagglehub
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

dataset_path = kagglehub.dataset_download("bakhtiyar2222/deep-sar-oil-spill-segmentation-refined")
print("Ścieżka do datasetu: ", dataset_path)

transform_rules = A.Compose([
    A.Resize(256, 256),        
    A.HorizontalFlip(p=0.5),   
    A.VerticalFlip(p=0.5),     
    A.RandomRotate90(p=0.5),   
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2(),              
])


class OilSpillDataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None):
        self.transform = transform
        self.image_dir = os.path.join(base_dir, "images", "images", split)
        self.mask_dir = os.path.join(base_dir, "masks", "masks", split)
                
        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Nie udało się znaleźć folderów treningowych. Sprawdź ścieżkę: {self.image_dir}")

        all_items = os.listdir(self.image_dir)
        self.images = sorted([
            f for f in all_items 
            if os.path.isfile(os.path.join(self.image_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"Znaleziono rzeczywiste obrazy treningowe: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.image_dir, img_name)
        
        mask_path = None
        for ext in ['.png', '.PNG', '.jpg', '.jpeg']:
            test_path = os.path.join(self.mask_dir, f"{base_name}{ext}")
            if os.path.exists(test_path):
                mask_path = test_path
                break
                
        if mask_path is None:
            mask = np.zeros((256, 256), dtype=np.float32)
        else:
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((256, 256), dtype=np.float32)
            else:
                mask = (mask / 255.0).astype(np.float32)  


        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask


def get_loaders(batch_size=8, test_size=0.5, random_state=42):
    train_ds  = OilSpillDataset(base_dir=dataset_path, split="train", transform=transform_rules)
    val_full  = OilSpillDataset(base_dir=dataset_path, split="val",   transform=None)

    val_files, test_files = train_test_split(
        val_full.images,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    val_ds  = OilSpillDataset(base_dir=dataset_path, split="val", transform=transform_rules)
    val_ds.images = val_files

    test_ds = OilSpillDataset(base_dir=dataset_path, split="val", transform=transform_rules)
    test_ds.images = test_files

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader