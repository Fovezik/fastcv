import cv2
import torch
import fastcv

img = cv2.imread("D:/Projects/Python/fastcv/artifacts/test.jpg")
img_tensor = torch.from_numpy(img).cuda()
bilateral_tensor = fastcv.bilateral_filter(img_tensor, diameter=15, sigma_color=75, sigma_space=75)
bilateral_image = bilateral_tensor.cpu().numpy()
cv2.imwrite("output_images/output_bilateral.jpg", bilateral_image)
print("saved bilateral filtered image.")
