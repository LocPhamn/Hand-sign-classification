import os.path
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as op
import numpy as np
from torchvision.transforms import Compose,ToTensor,Resize
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from MyHandSign import MyHandSign
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score , confusion_matrix
import cv2
import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--image-size","-i", type= int, default=224, help="Image Size")
    parse.add_argument("--checkpoint-dir","-c", type= str, default=r"D:\Python plus\AI_For_CV\CV_Project\Hand Sign with CNN\script\check_point", help="the path of best score and last score")
    parse.add_argument("--video-path","-o", type = str , default = "", help = "the path of video what is used for test", required = True)

    arg = parse.parse_args()
    return arg

def VideoTest():
    agrs = get_args()
    hand_move = ["Scale", "Point", "Other", "None", "Loupe", "Drag"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_path = agrs.video_path

    model = resnet18(weights= None)
    model.fc = nn.Linear(in_features=512, out_features=len(hand_move), bias=True)
    model.load_state_dict(torch.load(r"D:\Python plus\AI_For_CV\CV_Project\Hand Sign with CNN\script\check_point\best.pt",weights_only=True))
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    softmax = torch.nn.Softmax(dim=1)

    while cap.isOpened():
        ret, frame = cap.read()
        ori_image = frame
        ori_image = cv2.resize(ori_image, (500, 500))
        image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (agrs.image_size, agrs.image_size))
        image = image / 255.
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)

        model.eval()
        with torch.no_grad():
            image = image.to(device).float()
            output = model(image)
            output = softmax(output)
            predict, index = torch.max(output, 1)
            # predict = predict.tolist()
            text = "{}-{:.2f}%".format(hand_move[index], predict[0]*100)

        cv2.putText(ori_image,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255,0 ),2)
        cv2.imshow("image", ori_image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    VideoTest()