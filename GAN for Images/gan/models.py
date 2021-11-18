import torch
from gan.spectral_normalization import SpectralNorm

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.Lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=False)
        
        self.conv1 = torch.nn.Conv2d(3, 128, 4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 256, 4, stride=2, padding=1)
        
        self.bnc1 = torch.nn.BatchNorm2d(num_features=256)
        
        self.conv3 = torch.nn.Conv2d(256, 512, 4, stride=2, padding=1)
        
        self.bnc2 = torch.nn.BatchNorm2d(num_features=512)
        
        self.conv4 = torch.nn.Conv2d(512, 1024, 4, stride=2, padding=1)
        
        self.bnc3 = torch.nn.BatchNorm2d(num_features=1024)
        
        self.conv5 = torch.nn.Conv2d(1024, 1, 4, stride=1, padding=1)
        
        self.fc1 = torch.nn.Linear(1 * 3 * 3, 1)

        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.Lrelu(self.conv1(x))
        x = self.bnc1(self.Lrelu(self.conv2(x)))
        x = self.bnc2(self.Lrelu(self.conv3(x)))
        x = self.bnc3(self.Lrelu(self.conv4(x)))
        x = self.Lrelu(self.conv5(x))
        
        x = x.view(x.size()[0], 1 * 3 * 3)
        
        x = self.fc1(x)
        
        
        ##########       END      ##########
        
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.fc1 = torch.nn.Linear(self.noise_dim, 1024 * 4 * 4)
        
#         self.convT1 = torch.nn.ConvTranspose2d(self.noise_dim, 1024, 4, stride=2, padding=1)
        self.convT2 = torch.nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.convT3 = torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.convT4 = torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.convT5 = torch.nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)
        
        self.relu = torch.nn.ReLU(inplace=False)
        self.tanh = torch.nn.Tanh()
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.relu(self.fc1(x))
        x = x.view(x.size()[0], 1024, 4, 4)
#         x = torch.nn.Tanh(self.convT1(self.convT2(self.convT3(self.convT4(self.convT5(x))))))
#         x = self.convT1(x)
        x = self.relu(self.convT2(x))
        x = self.relu(self.convT3(x))
        x = self.relu(self.convT4(x))
        x = self.tanh(self.convT5(x))
        
        ##########       END      ##########
        
        return x
    
class Discriminator2(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator2, self).__init__()
        
        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.Lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=False)
        
        self.conv1 = torch.nn.Conv2d(3, 128, 4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 256, 4, stride=2, padding=1)
        
        self.bnc1 = torch.nn.BatchNorm2d(num_features=256)
        
        self.conv3 = torch.nn.Conv2d(256, 512, 4, stride=2, padding=1)
        
        self.bnc2 = torch.nn.BatchNorm2d(num_features=512)
        
        self.conv4 = torch.nn.Conv2d(512, 1024, 4, stride=2, padding=1)
        
        self.bnc3 = torch.nn.BatchNorm2d(num_features=1024)
        
        self.conv5 = torch.nn.Conv2d(1024, 1, 4, stride=1, padding=1)
        
        self.fc1 = torch.nn.Linear(1 * 7 * 7, 1)

        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.Lrelu(self.conv1(x))
        x = self.bnc1(self.Lrelu(self.conv2(x)))
        x = self.bnc2(self.Lrelu(self.conv3(x)))
        x = self.bnc3(self.Lrelu(self.conv4(x)))
        x = self.Lrelu(self.conv5(x))
        
        x = x.view(x.size()[0], 1 * 7 * 7)
        
        x = self.fc1(x)
        
        
        ##########       END      ##########
        
        return x


class Generator2(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator2, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.fc1 = torch.nn.Linear(self.noise_dim, 1024 * 8 * 8)
        
#         self.convT1 = torch.nn.ConvTranspose2d(self.noise_dim, 1024, 4, stride=2, padding=1)
        self.convT2 = torch.nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.convT3 = torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.convT4 = torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.convT5 = torch.nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)
        
        self.relu = torch.nn.ReLU(inplace=False)
        self.tanh = torch.nn.Tanh()
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.relu(self.fc1(x))
        x = x.view(x.size()[0], 1024, 8, 8)
#         x = torch.nn.Tanh(self.convT1(self.convT2(self.convT3(self.convT4(self.convT5(x))))))
#         x = self.convT1(x)
        x = self.relu(self.convT2(x))
        x = self.relu(self.convT3(x))
        x = self.relu(self.convT4(x))
        x = self.tanh(self.convT5(x))
        
        ##########       END      ##########
        
        return x
