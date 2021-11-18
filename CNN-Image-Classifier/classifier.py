import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Classifier(nn.Module):
    # TODO: implement me
    def __init__(self):
        super(Classifier, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.drop = nn.Dropout(p = 0.2)
        
        self.conv111 = nn.Conv2d(3, 32, 11, padding = 5)
        self.conv112 = nn.Conv2d(3, 17, 9, padding = 4)
        self.conv113 = nn.Conv2d(3, 10, 7, padding = 3)
        self.conv114 = nn.Conv2d(3, 4, 5, padding = 2)
        self.conv115 = nn.Conv2d(3, 1, 3, padding = 1)
        
        self.conv121 = nn.Conv2d(64, 10, 7, padding = 3)
        self.conv122 = nn.Conv2d(64, 20, 5, padding = 2)
        self.conv123 = nn.Conv2d(64, 34, 3, padding = 1)
        
        self.bnc1 = nn.BatchNorm2d(num_features=64)
        
        self.conv211 = nn.Conv2d(67, 96, 5, padding = 2)
        self.conv212 = nn.Conv2d(67, 32, 3, padding = 1)
        
        self.conv221 = nn.Conv2d(128, 32, 5, padding = 2)
        self.conv222 = nn.Conv2d(128, 96, 3, padding = 1)
        
        self.conv23 = nn.Conv2d(195, 128, 1)
        
        self.bnc2 = nn.BatchNorm2d(num_features=128)
        
        self.conv311 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv312 = nn.Conv2d(256, 128, 3, padding = 1)
        
        self.conv4 = nn.Conv2d(256, 256, 3, padding = 1)
        
        self.bnc3 = nn.BatchNorm2d(num_features=256)
        
        self.fc1 = nn.Linear(256 * 14 * 14, 1344)
        
        self.bn1 = nn.BatchNorm1d(num_features=1344)
        
        self.fc2 = nn.Linear(1344, 672)
        
        self.bn2 = nn.BatchNorm1d(num_features=672)
        
        self.fc3 = nn.Linear(672, 336)
        
        self.bn3 = nn.BatchNorm1d(num_features=336)
        
        self.fc4 = nn.Linear(336, 168)
        
        self.bn4 = nn.BatchNorm1d(num_features=168)
        
        self.fc5 = nn.Linear(168, 84)
        
        self.bn5 = nn.BatchNorm1d(num_features=84)
        
        self.fc6 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        Inception = self.pool(x)
        
        x = self.drop(F.relu(self.bnc1(torch.cat((self.conv111(x),self.conv112(x),self.conv113(x),self.conv114(x),self.conv115(x)),dim=1))))
        x = self.pool(F.relu(self.bnc1(torch.cat((self.conv121(x),self.conv122(x),self.conv123(x)),dim=1))))
        x = self.drop(torch.cat((x, Inception), dim = 1))
        
        Inception = self.pool(x)
        
        x = self.drop(F.relu(self.bnc2(torch.cat((self.conv211(x),self.conv212(x)),dim=1))))
        x = self.pool(F.relu(self.bnc2(torch.cat((self.conv221(x),self.conv222(x)),dim=1))))
        x = self.drop(torch.cat((x, Inception), dim = 1))
        
        x = F.relu(self.bnc2(self.conv23(x)))
        
        Inception = self.pool(x)
        
        x = self.drop(F.relu(self.bnc3(self.conv311(x))))
        x = self.pool(self.drop(F.relu(self.bnc2(self.conv312(x)))))
        x = self.drop(torch.cat((x, Inception), dim = 1))
        
        x = self.pool(self.drop(F.relu(self.bnc3(self.conv4(x)))))
        
        x = x.view(x.size()[0], 256 * 14 * 14)
        
        x = F.relu(self.bn1(self.fc1(x)))
        
        x = F.relu(self.bn2(self.fc2(x)))
        
        x = F.relu(self.bn3(self.fc3(x)))
        
        x = F.relu(self.bn4(self.fc4(x)))
                      
        x = F.relu(self.bn5(self.fc5(x)))
        
        x = self.fc6(x)
        return x
