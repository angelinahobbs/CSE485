import torch
import torch.nn as nn
import os
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, recall_score, precision_score

img_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# https://www.kaggle.com/competitions/dogs-vs-cats/data
img_root = "C:\\Users\\carso\\Downloads\\imageclassification\\dogs-vs-cats"
if not os.path.exists(img_root):
    raise FileNotFoundError(f"The path {img_root} does not exist.")

img_dataset = datasets.ImageFolder(root=img_root, transform=img_transform)

test_data, train_data = torch.utils.data.random_split(img_dataset, [int(0.8 * len(img_dataset)),
                                                                    len(img_dataset) - int(0.8 * len(img_dataset))])

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data)

hyperparams = {
    'lr': 0.001,
    'batch_size': 1,
    'epoch_size': 8
}

lr = hyperparams['lr']
batch_sz = hyperparams['batch_size']
epoch_sz = hyperparams['epoch_size']


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Conv2d(3, 16, 3, 1)
        self.layer2 = nn.Conv2d(16, 32, 3, 1)
        self.layer3 = nn.Conv2d(32, 64, 3, 1)
        self.layer4 = nn.Conv2d(64, 128, 3, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(1, -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=lr)

net.train()
for epoch in range(epoch_sz):

    total_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, label = data
        inputs = inputs.to(device)
        labels = label.to(device)

        opt.zero_grad()

        outputs = net(inputs)
        l = loss_fn(outputs, labels)
        l.backward()
        opt.step()

        total_loss += l.item()
        if i % 100 == 99:
            total_loss = 0.0

true_labels = []
pred_labels = []
net.eval()
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        true_labels += labels.tolist()
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        pred_labels += predicted.tolist()

acc = accuracy_score(true_labels, pred_labels)
rec = recall_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels)

print("accuracy score: " + str(acc))
print("recall score: " + str(rec))
print("precision score: " + str(prec))
