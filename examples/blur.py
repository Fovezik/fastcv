import os
import cv2
import torch
import fastcv

base_dir = os.path.dirname(os.path.abspath(__file__))
input_image_path = os.path.join(base_dir, "artifacts", "test.jpg")
output_image_path = os.path.join(base_dir, "output_images", "output_blur.jpg")

img = cv2.imread(input_image_path)
img_tensor = torch.from_numpy(img).cuda()
blurred_tensor = fastcv.blur(img_tensor, 10)
blurred_image = blurred_tensor.cpu().numpy()
cv2.imwrite(output_image_path, blurred_image)

print("saved blurred image.")