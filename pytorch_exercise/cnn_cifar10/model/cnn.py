import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class conv_net(nn.Module) :
    def __init__(self) :
        super(conv_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 36, 3, padding = 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(36, 64, 3, padding = 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 0)
        self.pool3 = nn.MaxPool2d(2, 2)
        # dropout
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*3*3, 276)
        self.fc2 = nn.Linear(276, 84)
        self.fc3 = nn.Linear(84, 10)
        # batch-normalization
        self.batch1 = nn.BatchNorm1d(num_features = 276)
        self.batch2 = nn.BatchNorm1d(num_features = 84)
        # weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.optimizer = optim.Adam(self.parameters(), lr = 0.0010)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x) :
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128*3*3)
        x = self.dropout3(x)
        x = F.relu(self.batch1(self.fc1(x)))
        x = F.relu(self.batch2(self.fc2(x)))
        x = self.dropout4(x)
        x = F.relu(self.fc3(x))
        return x

