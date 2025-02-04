import torch.nn as nn

# basic pytorch model
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(784, 64)
#         self.fc2 = nn.Linear(64, 10)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# model = Net()

#input: resume
#output: preference matrix

