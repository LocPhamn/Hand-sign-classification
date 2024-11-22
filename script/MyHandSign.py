from odbc import dataError
from torch.nn.parallel import data_parallel
from torchvision.transforms import Compose,ToTensor,Resize
from torch.utils.data import Dataset,DataLoader
import os
from  PIL import Image
import cv2

class MyHandSign(Dataset):
    def __init__(self,root,train,transform):
        self.hand_move = ["Scale","Point","Other","None","Loupe","Drag"]
        self.root = root
        self.images = []
        self.labels = []
        self.transform = transform
        # data_paths = ""
        if train:
            data_paths = os.path.join(root,"train")
        else:
            data_paths = os.path.join(root,"valid")
        for idx,hand in enumerate(self.hand_move):
            hand_paths = os.path.join(data_paths,hand)
            for image in os.listdir(hand_paths):
                image_path = os.path.join(hand_paths,image)
                self.images.append(image_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        # image = Image.open(self.images[item])
        image = cv2.imread(self.images[item],1)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        return image,label

# if __name__ == '__main__':
#     transform = Compose([
#         ToTensor(),
#         Resize((224,224))
#     ])
#     dataset = MyHandSign(root=r"D:\Python plus\AI_For_CV\dataset\hand_gestures_v2\dataset\real\images", train=True,transform=None)
#     handmove = dataset.hand_move
#     image,label = dataset[8500]
#     cv2.imshow(handmove[label],image)
#     cv2.waitKey(0)
    # image.show()
    # dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size=32,
    #     num_workers=8,
    #     shuffle=True,
    # )

    # for image,label in dataloader:
    #     print(image.shape)
