import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from Data_Spliting import CustomImageDataset
from torch.utils.data import DataLoader

# Hyper Parameters
Ladels = 3
EPOCH = 50             # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001             # learning rate
OPTIMIZER = "Adam"     # [SGD, Adagrad, RMSProp, Adam]
Feature_Filter = 32
Filter_Size = 5
Stride = 1
Padding = int(Filter_Size/2)
POOLING_SIZE = 2


# Load Data
train_data = CustomImageDataset(csv_file="train_data.csv", root_dir="All_gray_1_32_32", transform=transforms.ToTensor())
test_data = CustomImageDataset(csv_file="test_data.csv", root_dir="All_gray_1_32_32", transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)


# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(          # input shape (1, 32, 32)
            nn.Conv2d(
                in_channels=1,               # input height
                out_channels=Feature_Filter, # n_filters
                kernel_size=Filter_Size,     # filter size
                stride=Stride,               # filter movement/step
                padding=Padding,             # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                               # output shape (32, 32, 32)
            nn.ReLU(),                       # activation
            nn.MaxPool2d(kernel_size=POOLING_SIZE),    # choose max value in 2x2 area, output shape (32, 16, 16)
        )
        self.conv2 = nn.Sequential(          # input shape (32, 16, 16)
            nn.Conv2d(
                in_channels=Feature_Filter,  # input height
                out_channels=Feature_Filter*2, # n_filters
                kernel_size=Filter_Size,     # filter size
                stride=Stride,               # filter movement/step
                padding=Padding,
            ),                               # output shape (64, 16, 16)
            nn.ReLU(),                       # activation
            nn.MaxPool2d(kernel_size=POOLING_SIZE),    # output shape (64, 8, 8)
        )
        self.out = nn.Linear(64*8*8, Ladels)      # fully connected layer, output 3 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


# Check Accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            scores = model(x)[0]
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size()[0]
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)}/{float(num_samples)*1} = '
              f'{round((float(num_correct)/float(num_samples))*100, 2)}')

    model.train()


cnn = CNN()

# Optimizer Choice
if OPTIMIZER == "SGD":                                      # [SGD, Adagrad, RMSProp, Adam]
    optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)   # optimize all cnn parameters
elif OPTIMIZER == "Adagrad":
    optimizer = torch.optim.Adagrad(cnn.parameters(), lr=LR)
elif OPTIMIZER == "RMSProp":
    optimizer = torch.optim.RMSprop(cnn.parameters(), lr=LR)
elif OPTIMIZER == "Adam":
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

loss_func = nn.CrossEntropyLoss()                           # the target label is not one-hotted

# Training
list_loss = []
list_epoch =[]
for epoch in range(EPOCH):
    losses = []
    for step, (data, target) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

        # Forward
        output = cnn(data)[0]               # cnn output
        loss = loss_func(output, target)   # cross entropy loss
        losses.append(loss.item())

        # Backward
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients

        # Gradient Step
        optimizer.step()                # apply gradients

    list_loss.append(sum(losses)/len(losses))
    list_epoch.append(epoch)
    print(f'Loss at Epoch {epoch} is {sum(losses)/len(losses)}')

print("=================================================")
print("Epoch is", EPOCH)
print("-------------------------------------------------")
print("The batch size is", BATCH_SIZE)
print("-------------------------------------------------")
print("The number of feature filter is", Feature_Filter, ", size is", Filter_Size)
print("-------------------------------------------------")
print("The Pooling Size is", POOLING_SIZE)
print("-------------------------------------------------")
print("The optimizer is", OPTIMIZER)
print("-------------------------------------------------")
print("The loss function is CrossEntropyLoss")
print("-------------------------------------------------")
print("Check Accuracy of Training : ")
check_accuracy(train_loader, cnn)
print("-------------------------------------------------")
print("Check Accuracy of Testing : ")
check_accuracy(test_loader, cnn)

# Loss Graph
x = list_epoch
y = list_loss
plt.plot(x, y, 'bo-', linewidth=1.5)
plt.title("Loss Function")
plt.xlabel("EPOCH")
plt.ylabel("LOSS")
plt.grid(True)
plt.show()
