import os
import warnings
import torch
import torch.nn as nn
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from data import CoughVIDDataset
from model import CoughVIDModel

warnings.filterwarnings('ignore')


base_path = 'CoughVID/Images/covid/'
full_paths = []
for path in sorted(os.listdir(base_path)):
    full_paths.append(os.path.join(base_path, path))
base_path = 'CoughVID/Images/healthy/'
for path in sorted(os.listdir(base_path)):
    full_paths.append(os.path.join(base_path, path))

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


trainset = CoughVIDDataset(full_paths, transform=preprocess)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model = models.resnet50(pretrained=True)
for params in base_model.parameters():
    params.requires_grad = False

EPOCHS = 20
MFCC_SHAPE = (20, 500)

model = CoughVIDModel(base_model, MFCC_SHAPE)
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

for e in range(EPOCHS):
    correct = 0
    total = 0
    total_loss = 0
    for (img, mfcc), label in tqdm(trainloader):
        
        img, mfcc, label = img.to(device), mfcc.to(device), label.to(device)
        
        out = model(img, mfcc)
        
        loss = criterion(out, label[:, 0].long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            correct += (torch.argmax(out.data, 1) == label[:, 0]).sum().item()
            total += out.shape[0]
            total_loss += loss.cpu()
    print(f'Epoch {e}: Loss: {total_loss/total}, Accuracy: {correct/total*100}')
    torch.save(model, 'model.h5')
    scheduler.step()