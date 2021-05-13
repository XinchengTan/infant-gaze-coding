import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_img(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.bn2_2 = nn.BatchNorm2d(16)
        self.bn3_1 = nn.BatchNorm2d(32)
        self.bn3_2 = nn.BatchNorm2d(32)
        self.bn4_1 = nn.BatchNorm2d(64)
        self.bn4_2 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x))) + x
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x))) + x
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x))) + x
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


class Encoder_box(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 256)
        self.fc2 = nn.Linear(256, 256)
        self.bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(F.relu(self.bn(x)))
        x = self.fc2(x)

        return x


class Predictor_fc(nn.Module):
    def __init__(self, n, add_box):
        super().__init__()
        in_channel = 512 * n if add_box else 256 * n
        self.fc1 = nn.Linear(in_channel, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 3)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x.view(x.size(0), -1))
        x = self.dropout1(F.relu(self.bn1(x)))
        x = self.fc2(x)
        x = self.dropout2(F.relu(self.bn2(x)))
        x = self.fc3(x)
        return x


# class GazeCodingModel(nn.Module):
#     def __init__(self, device, n, add_box):
#         super().__init__()
#         self.n = n
#         self.device = device
#         self.add_box = add_box
#         self.encoder_img = Encoder_img()
#         self.encoder_box = Encoder_box()
#         self.predictor = Predictor_fc(n, add_box)
#
#     def forward(self, data):
#         imgs = data['imgs'].to(self.device)  # bs x n x 3 x 100 x 100
#         boxs = data['boxs'].to(self.device)  # bs x n x 5
#         embedding = self.encoder_img(imgs.view(-1, 3, 100, 100)).view(-1, self.n, 256)
#         if self.add_box:
#             box_embedding = self.encoder_box(boxs.view(-1, 5)).view(-1, self.n, 256)
#             embedding = torch.cat([embedding, box_embedding], -1)
#         pred = self.predictor(embedding)
#         return pred

from torchvision.models.resnet import resnet18

class GazeCodingModel(nn.Module):
    def __init__(self, device, n, add_box):
        super().__init__()
        self.n = n
        self.device = device
        self.add_box = add_box
        self.encoder_img = resnet18(num_classes=256)
        self.encoder_box = Encoder_box()
        self.predictor = Predictor_fc(n, add_box)

    def forward(self, data):
        imgs = data['imgs'].to(self.device)  # bs x n x 3 x 100 x 100
        boxs = data['boxs'].to(self.device)  # bs x n x 5
        embedding = self.encoder_img(imgs.view(-1, 3, 100, 100)).view(-1, self.n, 256)
        if self.add_box:
            box_embedding = self.encoder_box(boxs.view(-1, 5)).view(-1, self.n, 256)
            embedding = torch.cat([embedding, box_embedding], -1)
        pred = self.predictor(embedding)
        return pred