import os
import cv2
import torch
import fastcv

base_dir = os.path.dirname(os.path.abspath(__file__))
input_image_path = os.path.join(base_dir, "artifacts", "binary.jpg")
output_image_path = os.path.join(base_dir, "output_images", "output_dilated.jpg")

img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
img_tensor = torch.from_numpy(img).cuda()
gray_tensor = fastcv.dilate(img_tensor, 1)
gray_np = gray_tensor.squeeze(-1).cpu().numpy()
cv2.imwrite(output_image_path, gray_np)

print("saved dilated image.")
