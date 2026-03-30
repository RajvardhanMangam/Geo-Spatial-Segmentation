import rasterio
import matplotlib.pyplot as plt
import os
print(len(os.listdir("/home/ssl30/Desktop/geospace/data/processed/images")))
with rasterio.open("/home/ssl30/Desktop/geospace/data/processed/images/tile_4000.tif") as src:
    img = src.read().transpose(1,2,0)

with rasterio.open("/home/ssl30/Desktop/geospace/data/processed/masks/tile_4000.tif") as src:
    mask = src.read(1)

plt.subplot(1,2,1)
plt.imshow(img)

plt.subplot(1,2,2)
plt.imshow(mask)
plt.colorbar()

plt.show()

# import os
# import torch
# import rasterio
# import numpy as np
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from transformers import SegformerForSemanticSegmentation
# from tqdm import tqdm
# from torch.cuda.amp import autocast, GradScaler

# # =========================
# # CONFIG
# # =========================
# IMG_DIR = "data/processed/images"
# MASK_DIR = "data/processed/masks"

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BATCH_SIZE = 1
# EPOCHS = 100
# LR = 5e-5
# IMG_SIZE = 512   # VERY IMPORTANT

# # =========================
# # DATASET
# # =========================
# class GeoDataset(Dataset):
#     def __init__(self, img_dir, mask_dir):
#         self.files = sorted(os.listdir(img_dir))
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         name = self.files[idx]

#         # Load image
#         with rasterio.open(os.path.join(self.img_dir, name)) as src:
#             img = src.read().astype(np.float32) / 255.0

#         # Keep only RGB (drop 4th channel)
#         img = img[:3, :, :]

#         # Load mask
#         with rasterio.open(os.path.join(self.mask_dir, name)) as src:
#             mask = src.read(1).astype(np.int64)

#         # Remove utility class
#         mask[mask == 4] = 0

#         # Convert to torch
#         img = torch.tensor(img)
#         mask = torch.tensor(mask)

#         # Resize
#         img = F.interpolate(img.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='bilinear').squeeze(0)
#         mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
#                              size=(IMG_SIZE, IMG_SIZE),
#                              mode='nearest').squeeze(0).squeeze(0).long()

#         return img, mask

# # =========================
# # LOAD DATA
# # =========================
# dataset = GeoDataset(IMG_DIR, MASK_DIR)

# loader = DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=2,
#     pin_memory=True
# )

# # =========================
# # MODEL
# # =========================
# model = SegformerForSemanticSegmentation.from_pretrained(
#     "nvidia/segformer-b0-finetuned-ade-512-512",
#     num_labels=4,
#     ignore_mismatched_sizes=True
# ).to(DEVICE)

# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# scaler = GradScaler()

# # =========================
# # TRAIN LOOP
# # =========================
# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0

#     pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

#     for img, mask in pbar:
#         img = img.to(DEVICE)
#         mask = mask.to(DEVICE)

#         optimizer.zero_grad()

#         with autocast():
#             outputs = model(pixel_values=img, labels=mask)
#             loss = outputs.loss

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         total_loss += loss.item()
#         pbar.set_postfix({"loss": f"{loss.item():.4f}"})

#     print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(loader):.4f}")

#     torch.save(model.state_dict(), f"segformer_epoch_{epoch+1}.pth")