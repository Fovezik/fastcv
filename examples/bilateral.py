import os
import cv2
import torch
import fastcv

base_dir = os.path.dirname(os.path.abspath(__file__))
input_image_path = os.path.join(base_dir, "artifacts", "rubiks_cube.png")
output_image_path = os.path.join(base_dir, "output_images", "output_bilateral.jpg")

img = cv2.imread(input_image_path)
img_tensor = torch.from_numpy(img).cuda()
bilateral_tensor = fastcv.bilateral_filter(img_tensor, 15, 75, 75)
bilateral_image = bilateral_tensor.cpu().numpy()
cv2.imwrite(output_image_path, bilateral_image)
print("saved bilateral filtered image.")