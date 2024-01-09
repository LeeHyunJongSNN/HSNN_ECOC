import snntorch as snn
from snntorch import surrogate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from scipy.stats import bernoulli

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import argparse
import seaborn as sns

dtype = torch.float

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--data_path", type=str, default="propdata/MNIST")
parser.add_argument("--num_steps", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--ECOC", type=bool, default=True)
parser.add_argument("--FS", type=bool, default=True)
parser.add_argument("--FG", type=bool, default=False)
parser.add_argument("--FE", type=int, default=0)
parser.add_argument("--FL", type=int, default=2)
parser.add_argument("--FR", type=float, default=0.9)
parser.add_argument("--stuck", type=float, default=0.3)
parser.add_argument("--mod", type=bool, default=False)
parser.add_argument("--std", type=float, default=0.25)
parser.add_argument("--gpu_num", type=int, default=0)
parser.add_argument("--plot", type=bool, default=True)

args = parser.parse_args()

batch_size = args.batch_size
data_path = args.data_path
num_steps = args.num_steps
num_epochs = args.num_epochs
ECOC = args.ECOC
FS = args.FS
FG = args.FG
FE = args.FE
FL = args.FL
FR = args.FR
stuck = args.stuck
mod = args.mod
std = args.std
gpu_num = args.gpu_num
plot = args.plot

if gpu_num != 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("gpu: ", gpu_num)

# Define a transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

# Load MNIST/FashionMNIST dataset
if data_path == "propdata/MNIST":
    train_set = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)

elif data_path == "propdata/FashionMNIST":
    train_set = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

else:
    raise ValueError("Invalid data path!")

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

# Define the codes for ECOC
if ECOC:
    if data_path == "propdata/MNIST":
        defined_code = torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                     [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                                     [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                                     [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                                     [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
                                     [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
                                     [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
                                     [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0]], dtype=dtype).to(device)
        num_classes = len(defined_code)
        output_num = len(defined_code[0])

    elif data_path == "propdata/FashionMNIST":
        defined_code = torch.tensor([[0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                                     [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                     [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                                     [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                                     [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0],
                                     [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
                                     [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
                                     [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1]], dtype=dtype).to(device)
        num_classes = len(defined_code)
        output_num = len(defined_code[0])

    else:
        raise ValueError("Invalid data path!")

else:
    output_num = len(train_set.classes)

# Fault synapse simulation
if FL == 1:
    layer_size = (784, 1024)
elif FL == 2:
    layer_size = (1024, 512)
elif FL == 3:
    layer_size = (512, 128)
elif FL == 4:
    layer_size = (128, output_num)
elif FL == -1:
    layer_size = [(784, 1024), (1024, 512), (512, 128), (128, output_num)]
else:
    raise ValueError("Invalid layer number!")

FS_prev_num = int(layer_size[0] * FR)
FS_next_num = layer_size[1]

def SA0_1():
    if bernoulli.rvs(size=1, p=175/1079):
        return 1
    else:
        return -1

# Sudden
if FS:
    if FL == -1:
        FS_prev_num = []
        FS_next_num = []
        fault_mask = []
        fault_prev_pos = [[] for i in range(len(layer_size))]
        fault_next_pos = []
        for i in range(len(layer_size)):
            FS_prev_num.append(int(FR * layer_size[i][0]))
            FS_next_num.append(layer_size[i][1])

        for k in range(len(layer_size)):
            for i in range(FS_next_num[k]):
                fault_prev_pos[k].append(random.sample(range(0, layer_size[k][0]), FS_prev_num[k]))
            fault_next_pos.append(random.sample(range(0, layer_size[k][1]), FS_next_num[k]))
            fault_mask.append(torch.zeros_like(torch.zeros(layer_size[k]).to(device)))
            for i in range(len(fault_next_pos[k])):
                for j in fault_prev_pos[k][i]:
                    fault_mask[k][j, fault_next_pos[k][i]] = stuck * SA0_1()

    else:
        FS_prev_num = int(FR * layer_size[0])
        FS_next_num = layer_size[1]
        fault_prev_pos = []
        for i in range(FS_next_num):
            fault_prev_pos.append(random.sample(range(0, layer_size[0]), FS_prev_num))
        fault_next_pos = random.sample(range(0, layer_size[1]), FS_next_num)
        fault_mask = torch.zeros_like(torch.zeros(layer_size).to(device))
        for i in range(len(fault_next_pos)):
            for j in fault_prev_pos[i]:
                fault_mask[j, fault_next_pos[i]] = stuck * SA0_1()

# Define Network
spike_grad = surrogate.fast_sigmoid(slope=25)
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(784, 1024)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad)
        self.fc2 = nn.Linear(1024, 512)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad)
        self.fc3 = nn.Linear(512, 128)
        self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad)
        self.fc4 = nn.Linear(128, output_num)
        self.lif4 = snn.Leaky(beta=0.95, spike_grad=spike_grad, output=True)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        # Record the final layer
        spk4_rec = []
        mem4_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            spk4_rec.append(spk4)
            mem4_rec.append(mem4)

        return torch.stack(spk4_rec, dim=0), torch.stack(mem4_rec, dim=0)

# Load the network onto CUDA if available
net = Net().to(device)

# pass data into the network, sum the spikes over time
# and compare the neuron (index) with the highest number of spikes
# with the target

def code_convert(tensor: torch.Tensor):
    code_converted = []
    for i in range(tensor.size(0)):
        if tensor[i].item() == 0:
            code_converted.append(defined_code[0].detach().cpu().tolist())
        elif tensor[i].item() == 1:
            code_converted.append(defined_code[1].detach().cpu().tolist())
        elif tensor[i].item() == 2:
            code_converted.append(defined_code[2].detach().cpu().tolist())
        elif tensor[i].item() == 3:
            code_converted.append(defined_code[3].detach().cpu().tolist())
        elif tensor[i].item() == 4:
            code_converted.append(defined_code[4].detach().cpu().tolist())
        elif tensor[i].item() == 5:
            code_converted.append(defined_code[5].detach().cpu().tolist())
        elif tensor[i].item() == 6:
            code_converted.append(defined_code[6].detach().cpu().tolist())
        elif tensor[i].item() == 7:
            code_converted.append(defined_code[7].detach().cpu().tolist())
        elif tensor[i].item() == 8:
            code_converted.append(defined_code[8].detach().cpu().tolist())
        else:
            code_converted.append(defined_code[9].detach().cpu().tolist())

    return torch.tensor(code_converted).to(device)

def hamming_distance_pred(output: torch.Tensor, batch_size: int):
    bin_output = output.sum(dim=0).bool().int()
    pred = torch.zeros(batch_size, num_classes).to(device)
    for i in range(batch_size):
        for j in range(num_classes):
            ham_dist = distance.hamming(bin_output[i].detach().cpu().numpy(), defined_code[j].detach().cpu().numpy())
            pred[i, j] = ham_dist

    return pred.argmin(dim=1)

def print_batch_accuracy(data, targets, train=False):
    out_spk, _ = net(data.view(batch_size, -1))
    if ECOC:
        train_pred = hamming_distance_pred(out_spk, batch_size)
    else:
        _, train_pred = out_spk.sum(dim=0).max(1)
    acc = np.mean((targets == train_pred).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc * 100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc * 100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)
if ECOC:
    codes = code_convert(targets)
else:
    codes = None

spk_rec, mem_rec = net(data.view(batch_size, -1))

# initialize the loss function
loss_val = torch.zeros(1, dtype=dtype, device=device)
loss_fn = nn.CrossEntropyLoss()
for step in range(num_steps):
    if ECOC:
        loss_val += loss_fn(mem_rec[step], codes)
    else:
        loss_val += loss_fn(mem_rec[step], targets)

# printing the accuracy of the network without training
print_batch_accuracy(data, targets, train=True)

# clear previously stored gradients
optimizer.zero_grad()

# calculate the gradients
loss_val.backward()

# learning rate update
optimizer.step()

# calculate new network outputs using the same data
loss_val = torch.zeros(1, dtype=dtype, device=device)
for step in range(num_steps):
    if ECOC:
        loss_val += loss_fn(mem_rec[step], codes)
    else:
        loss_val += loss_fn(mem_rec[step], targets)

print(f"Training loss: {loss_val.item():.3f}")
print_batch_accuracy(data, targets, train=True)

loss_hist = []
test_loss_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)
    scheduler.step()

    # Gradual
    if FS and FG:
        if FL == -1:
            fault_mask = []
            for k in range(len(layer_size)):
                fault_mask.append(torch.zeros_like(torch.zeros(layer_size[k]).to(device)))
                size = int(len(fault_next_pos[k]) * (epoch - FE) / (num_epochs - FE - 1))
                for i in range(size):
                    for j in fault_prev_pos[k][i]:
                        fault_mask[k][j, fault_next_pos[k][i]] = stuck * SA0_1()

        else:
            fault_mask = torch.zeros_like(torch.zeros(layer_size)).to(device)
            size = int(len(fault_next_pos) * (epoch - FE) / (num_epochs - FE - 1))
            for i in range(size):
                for j in fault_prev_pos[i]:
                    fault_mask[j, fault_next_pos[i]] = stuck * SA0_1()


    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # Fault aware LR adjustment
        # if FS:
        #     if FG and epoch == num_epochs - 1:
        #         if FL == -1:
        #             optimizer = torch.optim.Adam(net.parameters(), lr=np.exp(FR) * 5e-3, betas=(0.9, 0.999))
        #         else:
        #             optimizer = torch.optim.Adam(net.parameters(), lr=np.exp(FR) * 5e-4, betas=(0.9, 0.999))
        #     else:
        #         if epoch > FE:
        #             if FL == -1:
        #                 optimizer = torch.optim.Adam(net.parameters(), lr=np.exp(FR) * 1e-3, betas=(0.9, 0.999))
        #             else:
        #                 optimizer = torch.optim.Adam(net.parameters(), lr=np.exp(FR) * 1e-4, betas=(0.9, 0.999))

        if ECOC:
            codes = code_convert(targets)
        else:
            codes = None

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros(1, dtype=dtype, device=device)
        for step in range(num_steps):
            if ECOC:
                loss_val += loss_fn(mem_rec[step], codes)
            else:
                loss_val += loss_fn(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():

            # Fault synapse
            if epoch > FE and FS:
                if FL == -1:
                    for i in range(len(layer_size)):
                        list(net.parameters())[i * 2].data *= torch.where(fault_mask[i].transpose(0, 1) != 0, 0,
                                                                          torch.ones_like(
                                                                              fault_mask[i].transpose(0, 1)))
                        if mod:
                            if stuck == 0:
                                fault_mask[i] = torch.randn_like(fault_mask[i]) * std + stuck
                            else:
                                fault_mask[i] = fault_mask[i].bool() * (torch.randn_like(fault_mask[i]) * std + stuck * SA0_1())

                        list(net.parameters())[i * 2].data += fault_mask[i].transpose(0, 1)

                else:
                    list(net.parameters())[(FL - 1) * 2].data *= torch.where(fault_mask.transpose(0, 1) != 0, 0,
                                                                             torch.ones_like(
                                                                                 fault_mask.transpose(0, 1)))
                    if mod:
                        if stuck == 0:
                            fault_mask = torch.randn_like(fault_mask) * std + stuck
                        else:
                            fault_mask = fault_mask.bool() * (torch.randn_like(fault_mask) * std + stuck * SA0_1())

                    list(net.parameters())[(FL - 1) * 2].data += fault_mask.transpose(0, 1)

            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)
            if ECOC:
                test_codes = code_convert(test_targets)
            else:
                test_codes = None

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(batch_size, -1))

            # Test set loss
            test_loss = torch.zeros(1, dtype=dtype, device=device)
            for step in range(num_steps):
                if ECOC:
                    test_loss += loss_fn(test_mem[step], test_codes)
                else:
                    test_loss += loss_fn(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer()
            counter += 1
            iter_counter += 1

layer_plot = []
layer_size_ = [(784, 1024), (1024, 512), (512, 128), (128, output_num)]
for i in range(len(layer_size_)):
    layer_plot.append(list(net.parameters())[i * 2].view(layer_size_[i][0], layer_size_[i][1]))

total = 0
correct = 0

# drop_last switched to False value to keep all samples
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)

w_targets = torch.tensor([], dtype=dtype).to(device)
w_predicted = torch.tensor([], dtype=dtype).to(device)

with torch.no_grad():
    net.eval()

    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        test_spk, test_mem = net(data.view(data.size(0), -1))

        # calculate total accuracy
        if ECOC:
            test_pred = hamming_distance_pred(test_spk, batch_size)
        else:
            _, test_pred = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (targets == test_pred).sum().item()
        w_targets = torch.cat((w_targets, targets), dim=0)
        w_predicted = torch.cat((w_predicted, test_pred), dim=0)

print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")

# Confusion Matrix
w_targets = w_targets.detach().cpu().numpy()
w_predicted = w_predicted.detach().cpu().numpy()

if plot:
    loss_fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(loss_hist)
    plt.plot(test_loss_hist)
    plt.title("Loss Curves")
    plt.legend(["Train Loss", "Test Loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    for i in range(len(layer_size_)):
        heat = plt.figure(facecolor="w", figsize=(8, 8))
        plt.imshow(layer_plot[i].detach().cpu().numpy(), cmap="hot", interpolation="nearest")
        plt.title(f"Layer {i + 1} to Layer {i + 2} Weights")
        plt.colorbar()
        plt.show()

    cm_fig = plt.figure(figsize=(8, 8))
    cm = confusion_matrix(w_targets, w_predicted)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    plt.title("Confusion Matrix")
    plt.show()
