import os
import cv2
import numpy as np
import kagglehub
import torch
from torch.utils.data import Dataset, DataLoader
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
        self.images = [
            f for f in all_items 
            if os.path.isfile(os.path.join(self.image_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.images = sorted(self.images)
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
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.0               
            mask = mask.astype(np.float32)   


        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask


def get_loaders(batch_size=8, split="train"): 
    dataset = OilSpillDataset(base_dir=dataset_path, split=split, transform=transform_rules)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
    return loader


if __name__ == "__main__":
    loader = get_loaders(batch_size=8, split="train")
    
    images_batch, masks_batch = next(iter(loader))
    print(f"X_batch: {images_batch.shape} | Y_batch: {masks_batch.shape} -> OK")