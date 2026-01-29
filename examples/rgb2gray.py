import os
import cv2
import torch
import fastcv

base_dir = os.path.dirname(os.path.abspath(__file__))
input_image_path = os.path.join(base_dir, "artifacts", "test.jpg")
output_image_path = os.path.join(base_dir, "output_images", "output_gray.jpg")

img = cv2.imread(input_image_path)
img_tensor = torch.from_numpy(img).cuda()
gray_tensor = fastcv.rgb2gray(img_tensor)
gray_np = gray_tensor.squeeze(-1).cpu().numpy()
cv2.imwrite(output_image_path, gray_np)

print("saved grayscale image.")
