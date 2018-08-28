import tqdm
import torch
import torchvision
import torchvision.transforms as transforms

mnist = torchvision.datasets.MNIST('./MNIST',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))]))
loader = torch.utils.data.DataLoader(mnist,batch_size=16,shuffle=True,num_workers=4)
for i,data in tqdm.tqdm(enumerate(loader,1)):
    img,_ = data
    torchvision.utils.save_image(img,'./RealImages/%d.png' % i,nrow=8)
