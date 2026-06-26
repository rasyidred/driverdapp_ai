import cv2
import torch
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
import os
from glob import glob

# ----------------------------
# Load YOLO Classification Model
# ----------------------------
yolo = YOLO("classification_model.pt")
model = yolo.model.model  # Pure PyTorch classification network
model.eval()

# ----------------------------
# Find Last Conv Layer
# ----------------------------
def find_last_conv(net):
    last = None
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2D layer found in YOLO classification model")
    return last

# ----------------------------
# Grad-CAM implementation
# ----------------------------
def generate_gradcam(net, input_tensor, target_class=None):
    net.eval()
    conv_out = None
    grad_out = None

    def fwd_hook(m, inp, out):
        nonlocal conv_out
        conv_out = out

    def bwd_hook(m, gin, gout):
        nonlocal grad_out
        grad_out = gout[0]

    last_conv = find_last_conv(net)
    h1 = last_conv.register_forward_hook(fwd_hook)
    h2 = last_conv.register_backward_hook(bwd_hook)

    input_tensor.requires_grad = True
    logits = net(input_tensor)          # Now gradients enabled

    if isinstance(logits, tuple):  # Check if the output is a tuple (common in YOLO)
        logits = logits[0]  # Extract logits tensor

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    net.zero_grad()
    logits[0, target_class].backward()  # Perform backpropagation for Grad-CAM

    h1.remove()
    h2.remove()

    # Compute Grad-CAM
    weights = torch.mean(grad_out, dim=(2, 3), keepdim=True)
    cam = F.relu(torch.sum(weights * conv_out, dim=1)).squeeze()
    cam = cam - cam.min()
    cam = cam / cam.max()

    return cam.detach().cpu().numpy()

# ----------------------------
# Input/output folders
# ----------------------------
IMAGE_DIR = "./PATH_TO_DATASET_IMAGES"
OUTPUT_DIR = "./output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = glob(os.path.join(IMAGE_DIR, "*.*"))
print("Found images:")
for p in image_paths:
    print(" -", p)

if len(image_paths) == 0:
    print("ERROR: No images found!")
    exit()

# ----------------------------
# Process images
# ----------------------------
for path in image_paths:
    print(f"\nProcessing {path}")

    img = cv2.imread(path)
    if img is None:
        print("ERROR reading image")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(img_rgb, (320, 320))
    chw = np.transpose(resized, (2, 0, 1)) / 255.0
    tensor = torch.tensor(chw).unsqueeze(0).float()

    # ----- Get probability prediction -----
    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits, dim=1)
        top_class = probs.argmax(dim=1).item()
        confidence = float(probs[0, top_class])
        class_name = yolo.names[top_class]

    # ----- Generate Grad-CAM -----
    cam = generate_gradcam(model, tensor.clone(), target_class=top_class)
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # ----- Annotate -----
    cv2.putText(overlay, f"{class_name} ({confidence:.2f})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

    # ----- Save -----
    save_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
    ok = cv2.imwrite(save_path, overlay)

    if ok:
        print(f"Saved: {save_path}")
    else:
        print(f"ERROR saving file")

print("\nDONE!")
