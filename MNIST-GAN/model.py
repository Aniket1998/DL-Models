import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

#Generate an image from a 128*1 vector sampled from the noise prior


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 28*28)
        self.bn4 = nn.BatchNorm1d(28*28)
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
            torch.nn.init.xavier_uniform_(self.fc4.weight,gain=4)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = F.sigmoid(self.bn4(self.fc4(x))) #Images are in the range (0,1) hence so should be the generator outputs
        x = x.reshape(-1,28,28)
        return x.unsqueeze(1)

##Instead of using a scalar for classification: i.e. 0 for fake and 1 for real, most popular
#implementations use a vector of a given size n where [0]n is for fake and [1]n is for real
#We use n=128


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(p=0.3)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = x.reshape(-1,28*28)
        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.leaky_relu(self.bn3(self.fc3(x))))
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        return x
