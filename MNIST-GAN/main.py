import model
import trainer
import torch
import torchvision
import torchvision.transforms as transforms

mnist = torchvision.datasets.MNIST('./MNIST',download=True,transform=transforms.ToTensor())
generator = model.Generator()
discriminator = model.Discriminator()
device = torch.device('cuda:0')
Trainer = trainer.GANTrainer(device,G=generator,D=discriminator,dataset=mnist,batch_size=32,epochs=200,lr=0.0002,model='./GAN.model',images='./Images')
#Trainer.load_model()
Trainer.train()
