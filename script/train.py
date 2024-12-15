import os.path

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
    parse.add_argument("--epochs", "-e", type=int, default=15, help="the number of epochs")
    parse.add_argument("--batch-size", "-b", type=int, default=32, help="the number of batch")
    parse.add_argument("--num-workers","-n", type=int, default=8, help="the number of core flow")
    parse.add_argument("--image-size","-i", type= int, default=224, help="Image Size")
    parse.add_argument("--checkpoint-dir","-c", type= str, default=r"D:\Python plus\AI_For_CV\CV_Project\Hand Sign with CNN\script\check_point", help="the path of best score and last score")
    parse.add_argument("--image-path","-p", type = str , default = "", help = "the path of image what is used for test", required = True)

    arg = parse.parse_args()
    return arg

def train():
    agrs = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = agrs.epochs
    batch_size = agrs.batch_size
    nums_worker = agrs.num_workers
    check_point_dir = agrs.checkpoint_dir
    transform = Compose([
        ToTensor(),
        Resize((agrs.image_size,agrs.image_size))
    ])

    train_dataset = MyHandSign(root=r"D:\Python plus\AI_For_CV\dataset\hand_gestures_v2\dataset\real\images",train=True,transform=transform)
    train_loader = DataLoader(
        dataset= train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers =nums_worker)
    test_dataset = MyHandSign(root=r"D:\Python plus\AI_For_CV\dataset\hand_gestures_v2\dataset\real\images", train=False,transform=transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=nums_worker)

    model = resnet18(weights= ResNet18_Weights)
    model.fc = nn.Linear(in_features=512, out_features=len(train_dataset.hand_move), bias=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = op.Adam(model.parameters(), lr=0.001)

    if not os.path.isdir(check_point_dir):
        os.mkdir(check_point_dir)

    best_accuracy = 0
    min_loss = 10

    #QUÁ TRÌNH TRAIN
    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_accuracy = []
        progress_bar = tqdm(train_loader,colour= "yellow")
        for img,label in progress_bar:
            # Forward
            image = img.to(device)
            label = label.to(device)
            output = model(image)
            loss_score = criterion(output, label)
            predict = torch.argmax(output, 1)
            train_loss.append(loss_score.item())
            train_accuracy.append(accuracy_score(label.tolist(), predict.tolist()))
            #Backward
            optimizer.zero_grad()
            loss_score.backward()
            optimizer.step()
            progress_bar.set_description("epochs {}/{} loss: {:0.4f}, accuracy: {}".format(epoch + 1, epochs, loss_score.item(),accuracy_score(label.tolist(), predict.tolist())))
        print("mean of loss_score: {}".format(np.mean(train_loss)))



        # QUÁ TRÌNH VALID
        valid_labels = []
        valid_predictions = []
        valid_loss = []
        model.eval()
        with torch.no_grad():
            for image, label in test_loader:
                image = image.to(device)
                label = label.to(device)
                output = model(image)
                loss_score = criterion(output, label)
                predict = torch.argmax(output, 1)

                valid_labels.extend(label.tolist())
                valid_predictions.extend(predict.tolist())
                valid_loss.append(loss_score.item())
            valid_accuracy = accuracy_score(valid_labels,valid_predictions)
            avg_loss = np.mean(valid_loss)
            if best_accuracy < valid_accuracy and min_loss > avg_loss:
                best_accuracy = valid_accuracy
                min_loss = avg_loss
                torch.save(model.state_dict(),os.path.join(check_point_dir,"best.pt"))
            print(valid_accuracy)


def test():
    agrs = get_args()
    hand_move = ["Scale", "Point", "Other", "None", "Loupe", "Drag"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = resnet18(weights= None)
    model.fc = nn.Linear(in_features=512, out_features=6, bias=True)
    model.to(device)
    model.load_state_dict(torch.load(r"D:\Python plus\AI_For_CV\CV_Project\Hand Sign with CNN\script\check_point\best.pt",weights_only=True))

    ori_image = cv2.imread(agrs.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image)
    n = torch.nn.Softmax()

    model.eval()
    with torch.no_grad():
        image = image.to(device).float()
        output = model(image)
        output = n(output)
        # predict, index = torch.max(output, 1)
        predict, index = torch.max(output, 1)
        # predict = predict.tolist()
        ori_image = cv2.resize(ori_image,(500,500))
        cv2.imshow("{}-{}".format(hand_move[index],predict[0]*100), ori_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    train()
    test()
