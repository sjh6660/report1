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


trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=32, num_classes=10):
        super().__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(3 * 32, 128)
        self.pos_emb = nn.Parameter(torch.randn(1, 32, 128))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):

        x = x.view(x.size(0), 32, 3 * 32)
        x = self.embedding(x)
        x = x + self.pos_emb
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(trainloader):.3f}")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

print(f"Transformer 测试准确率: {100 * correct / total:.2f}%")