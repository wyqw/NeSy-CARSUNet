import cv2
import numpy as np

import os
import torch
from torchvision import transforms
from PIL import Image

from src.models.CARSUNet import CARSUNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def segmentlb(imgpath):

    weights_path = "save_weights/example.pth"
    img_path = imgpath

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = CARSUNet(num_classes=3, pretrain_backbone=False)

    weights_dict = torch.load(weights_path, map_location='cpu',weights_only=False)['model']
    model.load_state_dict(weights_dict)
    model.to(device)

    data_transform = transforms.Compose([
        transforms.Resize(565),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    original_img = Image.open(img_path).convert("RGB")
    img = data_transform(original_img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img)
        prediction = output['out'].argmax(1).squeeze(0).to("cpu").numpy()

    pred_resize = Image.fromarray(prediction.astype(np.uint8)).resize(original_img.size, Image.NEAREST)
    pred_resize_np = np.array(pred_resize)

    result_img_np = np.zeros((original_img.height, original_img.width, 3), dtype=np.uint8)
    original_img_np = np.array(original_img)

    result_img_np[pred_resize_np == 1] = [0, 128, 0]
    result_img_np[pred_resize_np == 2] = [128, 0, 0]
    result_img1 = Image.fromarray(result_img_np)

    result_img1.save("./predict.png")

def gradinglb():
    image = cv2.imread('predict.png')

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    leafmask = cv2.inRange(image, (0, 128, 0), (0, 128, 0))
    green_pixels = cv2.countNonZero(leafmask)
    print("The number of pixels in the leaf region: ", green_pixels)

    lower_red = np.array([0, 100, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    red_pixels = cv2.countNonZero(mask)
    print("The number of pixels in the lesion: ", red_pixels)
    ratio1 = "{:.4f}".format(red_pixels / (green_pixels + red_pixels))

    if float(ratio1) > float(0.5):
        print("Late Blight Level: 9. (Ratio of lesions: {})".format(ratio1))
    elif float(ratio1) > float(0.3):
        print("Late Blight Level: 7. (Ratio of lesions: {})".format(ratio1))
    elif float(ratio1) > float(0.15):
        print("Late Blight Level: 5. (Ratio of lesions: {})".format(ratio1))
    elif float(ratio1) > float(0.05):
        print("Late Blight Level: 3. (Ratio of lesions: {})".format(ratio1))
    elif float(ratio1) > float(0.0):
        print("Late Blight Level: 1. (Ratio of lesions: {})".format(ratio1))
    else:
        print("Healthy Leaf.)")

if __name__ == '__main__':
    test_img = " "   # path
    segmentlb(imgpath=test_img)
    gradinglb()
