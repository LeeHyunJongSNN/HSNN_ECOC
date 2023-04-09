import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import minmax_scale
from scipy.stats import bernoulli
from keras.datasets import mnist

import matplotlib.pyplot as plt
import numpy as np
import itertools
import random

# dataloader arguments
batch_size = 128
data_path = 'propdata/MNIST'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# Network Architecture
num_inputs = 28 * 28
num_hidden = 1000
num_outputs = 10

# Temporal Dynamics
num_steps = 25
beta = 0.95
# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# Load the network onto CUDA if available
net = Net().to(device)

# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)
spk_rec, mem_rec = net(data.view(batch_size, -1))
print(mem_rec.size())

# initialize the total loss value
loss_val = torch.zeros(1, dtype=dtype, device=device)

# sum loss at every step
for step in range(num_steps):
    loss_val += loss(mem_rec[step], targets)
print(f"Training loss: {loss_val.item():.3f}")
print_batch_accuracy(data, targets, train=True)

# clear previously stored gradients
optimizer.zero_grad()

# calculate the gradients
loss_val.backward()

# weight update
optimizer.step()

# calculate new network outputs using the same data
spk_rec, mem_rec = net(data.view(batch_size, -1))

# initialize the total loss value
loss_val = torch.zeros(1, dtype=dtype, device=device)

# sum loss at every step
for step in range(num_steps):
    loss_val += loss(mem_rec[step], targets)

print(f"Training loss: {loss_val.item():.3f}")
print_batch_accuracy(data, targets, train=True)

num_epochs = 1
loss_hist = []
test_loss_hist = []
counter = 0

ST = True
AST = False
drop_num = 0        # AST 50
reinforce_num = 1
num_AST = num_hidden
pre_average = []
dropout_index = []
drop_input = []
reinforce_input = []
reinforce_ref = []
dead_input = []

pre = mnist.load_data()
pre_x = pre[0][0].reshape(60000, 784)
pre_y = pre[0][1].reshape(60000, 1)
preprocessed = np.concatenate((pre_x, pre_y), axis=1)
preprocessed = preprocessed[preprocessed[:, 784].argsort()]
preprocessed = preprocessed[:, 0:784]

num_inputs = 784
pre_size = 6000
entire = np.sort(np.mean(preprocessed, axis=0))

if ST:
    for i in range(10):
        pre_average.append(np.mean(preprocessed[i * pre_size:(i + 1) * pre_size], axis=0))

        if AST:
            drop_num = len(np.where(pre_average[i] <= entire[int(num_inputs * 0.3) - 1])[0])
            reinforce_num = len(np.where(pre_average[i] >= entire[int(num_inputs) - 1])[0])

        drop_input.append(np.argwhere(pre_average[i] < np.sort(pre_average[i])[0:drop_num + 1][-1]).flatten())
        reinforce_input.append(
            np.argwhere(pre_average[i] > np.sort(pre_average[i])[0:num_inputs - reinforce_num][-1]).flatten())
        if reinforce_num != 0:
            values = np.sort(pre_average[i])[::-1][:reinforce_num]
            reinforce_ref.append(values / np.max(values))
        else:
            reinforce_ref.append([])

drop_input *= int(np.ceil(num_hidden / 10))
reinforce_input *= int(np.ceil(num_hidden / 10))
reinforce_ref *= int(np.ceil(num_hidden / 10))
template_exc = np.arange(num_hidden)

ADC = False

DS = True
DS_out_num = 10
DS_hidden_num = 800
dead_input = []
if DS:
    for i in range(DS_out_num):
        dead_input.append(random.sample(range(0, num_hidden), DS_hidden_num))
dead_out = random.sample(range(0, num_outputs), DS_out_num)

SC = False

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros(1, dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # Synaptic Template
        if ST:
            for i in range(len(template_exc)):
                for j in drop_input[i]:
                    a = list(net.parameters())[0].data
                    list(net.parameters())[0].data[template_exc[i], j] = 0
                for j in reinforce_input[i]:
                    if list(net.parameters())[0].data[template_exc[i], j] <= 1:
                        list(net.parameters())[0].data[template_exc[i], j] = \
                            reinforce_ref[int(template_exc[i])][int(np.where(j == reinforce_input[i])[0])] * 0.2

        # Dead synapses
        if DS:
            for i in range(len(dead_out)):
                for j in dead_input[i]:
                    list(net.parameters())[2].data[dead_out[i], j] = 0

        # Synaptic Weight Scaling
        if SC:
            list(net.parameters())[2].data *= 1.0001     # 1.0001

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Adaptive Drop Connect
        if ADC:
            p = np.round(minmax_scale(
                np.nan_to_num(list(net.parameters())[2].data.cpu().detach().numpy().reshape(num_outputs * num_hidden),
                              copy=False),
                feature_range=(0.9995, 1)).reshape(num_outputs, num_hidden), 3)
            m = torch.zeros(num_outputs, num_hidden).to('cuda')

            for i in range(num_outputs):
                for j in range(num_hidden):
                    m[i, j] = int(bernoulli.rvs(p[i, j], size=1))

            list(net.parameters())[2].data *= m

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(batch_size, -1))

            # Test set loss
            test_loss = torch.zeros(1, dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer()
            counter += 1
            iter_counter += 1

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

total = 0
correct = 0

# drop_last switched to False value to keep all samples
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
    net.eval()
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        test_spk, _ = net(data.view(data.size(0), -1))

        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
