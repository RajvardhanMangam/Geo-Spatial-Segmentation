import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

# =========================
# CONFIG
# =========================
MODEL_PATH = "/home/ssl30/Desktop/geospace/segformer_epoch_100.pth"   # change if needed
IMAGE_PATH = "data/processed/images/tile_4000.tif"
with rasterio.open("/home/ssl30/Desktop/geospace/data/processed/masks/tile_4000.tif") as src:
    mask = src.read(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512

# =========================
# LOAD MODEL
# =========================
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=4,
    ignore_mismatched_sizes=True
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# =========================
# LOAD IMAGE
# =========================
with rasterio.open(IMAGE_PATH) as src:
    img = src.read().astype(np.float32) / 255.0

# keep RGB only
img = img[:3, :, :]

# convert to tensor
img_tensor = torch.tensor(img)

# resize
img_tensor = F.interpolate(
    img_tensor.unsqueeze(0),
    size=(IMG_SIZE, IMG_SIZE),
    mode='bilinear'
).to(DEVICE)

# =========================
# PREDICTION
# =========================
with torch.no_grad():
    outputs = model(pixel_values=img_tensor)
    logits = outputs.logits

# resize back to original
logits = F.interpolate(logits, size=img.shape[1:], mode='bilinear')

pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

# =========================
# VISUALIZATION
# =========================
img_show = img.transpose(1, 2, 0)

plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.title("Image")
plt.imshow(img_show)

plt.subplot(1,3,2)
plt.title("Prediction")
plt.imshow(pred_mask)
plt.colorbar()
plt.subplot(1,3,3)
plt.title("Ground truth")
plt.imshow(mask)
plt.colorbar()

plt.show()