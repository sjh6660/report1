import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


DATA_ROOT = 'D:/datasets/cifar10'


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


trainset = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True
)

testset = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=False, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False
)



class LSTM(nn.Module):
    def __init__(self, input_size=32, hidden_size=128, num_layers=2, num_classes=10):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)


        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = x.view(x.size(0), 3 * 32, 32)


        out, _ = self.lstm(x)

        out = out[:, -1, :]


        out = self.fc(out)
        return out



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/10], Loss: {total_loss / len(trainloader):.4f}")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"LSTM 测试准确率: {100 * correct / total:.2f}%")