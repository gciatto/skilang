from torch import nn, optim

from skilang.usecase.poker.dataset import train_loader, test_loader
from skilang.usecase.poker.model import model


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.fit(train_loader, test_loader, criterion, optimizer, epochs=2)
model.plot_loss()
model.save("model.pth")
