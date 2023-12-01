import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.spatial import distance
from bindsnet.utils import get_square_weights
from sklearn.metrics import confusion_matrix

# import matplotlib
# matplotlib.use("TkAgg")
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
parser.add_argument("--code_type", type=str, default=None)
parser.add_argument("--FS", type=str, default="train")
parser.add_argument("--FT", type=str, default="sporadic")
parser.add_argument("--FG", type=bool, default=False)
parser.add_argument("--FE", type=int, default=0)
parser.add_argument("--FL", type=int, default=2)
parser.add_argument("--FR", type=float, default=0.1)
parser.add_argument("--stuck", type=float, default=0.3)
parser.add_argument("--mod", type=bool, default=False)
parser.add_argument("--std", type=float, default=0.5)
parser.add_argument("--gpu_num", type=int, default=0)
parser.add_argument("--plot", type=bool, default=False)

args = parser.parse_args()

batch_size = args.batch_size
data_path = args.data_path
num_steps = args.num_steps
num_epochs = args.num_epochs
code_type = args.code_type
FS = args.FS
FT = args.FT
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

# Define codes for ECOC
if FS is None:
    scale = 1.0
else:
    if 0.1:
        scale = 1.0
    elif FR == 0.2:
        scale = 0.8
    elif FR == 0.3:
        scale = 0.4
    elif 0.4 <= FR <= 0.6:
        scale = 0.2
    else:
        scale = 0.1

if data_path == "propdata/MNIST":

    if code_type is None:
        output_num = len(train_set.classes)

    elif code_type == "bin":
        defined_code = torch.tensor([[1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0],
                                     [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
                                     [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
                                     [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                                     [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                                     [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                     [1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                                     [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
                                     [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=dtype).to(device)
        defined_code *= scale
        num_classes = len(defined_code)
        output_num = len(defined_code[0])

    elif code_type == "mul":
        defined_code = torch.tensor([[1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0],
                                     [1, 1, 0, 0, 1, 1, 0, 0, 2, 1, 1, 1],
                                     [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
                                     [2, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
                                     [0, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 1],
                                     [1, 0, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                     [1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                                     [1, 0, 1, 1, 1, 0, 0, 1, 3, 0, 1, 1],
                                     [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 3, 0],
                                     [1, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0]], dtype=dtype).to(device)
        defined_code *= scale
        num_classes = len(defined_code)
        output_num = len(defined_code[0])

    else:
        raise ValueError("Invalid code type!")

elif data_path == "propdata/FashionMNIST":

    if code_type is None:
        output_num = len(train_set.classes)

    elif code_type == "bin":
        defined_code = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
                                     [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                     [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                                     [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                     [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
                                     [1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                     [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                     [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                                     [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]], dtype=dtype).to(device)
        defined_code *= scale
        num_classes = len(defined_code)
        output_num = len(defined_code[0])

    elif code_type == "mul":
        defined_code = torch.tensor([[2, 1, 1, 1, 1, 1, 1, 0, 1, 3, 1, 1, 0, 2, 0],
                                     [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                     [0, 1, 1, 1, 1, 1, 2, 1, 0, 1, 0, 0, 1, 3, 2],
                                     [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                     [1, 1, 1, 1, 3, 0, 0, 0, 1, 1, 0, 1, 2, 1, 3],
                                     [1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 3, 3, 3],
                                     [1, 0, 0, 0, 2, 0, 1, 1, 1, 1, 1, 1, 0, 2, 3],
                                     [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                                     [1, 3, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 3, 1, 2]], dtype=dtype).to(device)
        defined_code *= scale
        num_classes = len(defined_code)
        output_num = len(defined_code[0])

    else:
        raise ValueError("Invalid code type!")

else:
    raise ValueError("Invalid data path!")

multi_defined = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 1, 1, 0, 0, 0, 1, 1],
                              [0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
                              [1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                              [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
                              [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                              [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                              [0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
                              [0, 0, 1, 1, 1, 1, 1, 0, 0, 1],
                              [0, 0, 1, 0, 1, 1, 1, 1, 0, 1]], dtype=dtype).to(device)

# Fault synapse simulation
if FL == 1:
    layer_size = (784, 1024)
elif FL == 2:
    layer_size = (1024, 512)
elif FL == 3:
    layer_size = (512, 128)
elif FL == 4:
    layer_size = (128, output_num)
else:
    raise ValueError("Invalid layer number!")

FS_prev_num = int(layer_size[0] * FR)
FS_next_num = layer_size[1]

if FS == "train" or FS == "test":
    if FT == "sporadic":
        fault_prev_pos = []
        for i in range(FS_next_num):
            fault_prev_pos.append(random.sample(range(0, layer_size[0]), FS_prev_num))
        fault_next_pos = random.sample(range(0, layer_size[1]), FS_next_num)
        fault_mask = torch.zeros_like(torch.zeros(layer_size).to(device))
        for i in range(len(fault_next_pos)):
            for j in fault_prev_pos[i]:
                fault_mask[j, fault_next_pos[i]] = stuck

    elif FT == "linear":
        fault_mask = torch.zeros_like(torch.zeros(layer_size).to(device))
        fault_prev_pos = random.sample(range(0, layer_size[0]), FS_prev_num)
        fault_next_pos = random.sample(range(0, layer_size[1]), FS_next_num)
        for i in fault_prev_pos:
            for j in fault_next_pos:
                fault_mask[i, j] = stuck

    elif FT == "arial":
        fault_mask = torch.zeros_like(torch.zeros(layer_size).to(device))
        prev_start = random.randint(0, layer_size[0] - FS_prev_num)
        next_start = random.randint(0, layer_size[1] - FS_next_num)
        for i in range(prev_start, prev_start + FS_prev_num):
            for j in range(next_start, next_start + FS_next_num):
                fault_mask[i, j] = stuck

    else:
        raise ValueError("Invalid fault type!")

elif FS is None:
    fault_mask = torch.ones_like(torch.zeros(layer_size)).to(device)

else:
    raise ValueError("Invalid fault simulation type!")

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, output_num)

    def forward(self, x):

        x = self.fc1(x)
        x = nn.functional.elu(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        x = nn.functional.elu(x)
        x = self.fc3(x)
        x = nn.functional.elu(x)
        x = self.fc4(x)
        if code_type is None:
            x = nn.functional.log_softmax(x, dim=1)
        else:
            x = nn.functional.elu(x)

        return x

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

def multi_convert(multi_code: int):
    if multi_code == 0:
        multi_converted = multi_defined[0]
    elif multi_code == 1:
        multi_converted = multi_defined[1]
    elif multi_code == 2:
        multi_converted = multi_defined[2]
    elif multi_code == 3:
        multi_converted = multi_defined[3]
    elif multi_code == 4:
        multi_converted = multi_defined[4]
    elif multi_code == 5:
        multi_converted = multi_defined[5]
    elif multi_code == 6:
        multi_converted = multi_defined[6]
    elif multi_code == 7:
        multi_converted = multi_defined[7]
    elif multi_code == 8:
        multi_converted = multi_defined[8]
    else:
        multi_converted = multi_defined[9]

    return torch.tensor(multi_converted, dtype=dtype).to(device)

def bin_distance_pred(output: torch.Tensor, batch_size: int):
    bin_output = (torch.clip(output, min=0) + 0.005).round(decimals=1)
    if scale.is_integer():
        bin_output = (torch.clip(output, min=0) + 0.05).round()
    pred = torch.zeros(batch_size, num_classes).to(device)
    for i in range(batch_size):
        for j in range(num_classes):
            ham_dist = distance.hamming(bin_output[i].detach().cpu().numpy(), defined_code[j].detach().cpu().numpy())
            pred[i, j] = ham_dist

    return pred.argmin(dim=1)

def mul_distance_pred(output: torch.Tensor, batch_size: int):
    bin_output = (torch.clip(output, min=0, max=num_classes - 1) + 0.005).round(decimals=1)
    if scale.is_integer():
        bin_output = (torch.clip(output, min=0, max=num_classes - 1) + 0.05).round()
    pred = torch.zeros(batch_size, num_classes).to(device)
    for i in range(batch_size):
        for j in range(num_classes):
            for k in range(output_num):
                ham_dist = distance.hamming(multi_convert(bin_output[i, k]).detach().cpu().numpy(),
                                            multi_convert(defined_code[j, k]).detach().cpu().numpy())
                pred[i, j] += ham_dist

    return pred.argmin(dim=1)

def print_batch_accuracy(data, targets, train=False):
    output = net(data.view(batch_size, -1))
    if code_type == "bin":
        train_pred = bin_distance_pred(output, batch_size)
    elif code_type == "mul":
        train_pred = mul_distance_pred(output, batch_size)
    else:
        train_pred = output.argmax(dim=1)
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

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)
if code_type is None:
    codes = None
else:
    codes = code_convert(targets)

output = net(data.view(batch_size, -1))

# initialize the loss function
if code_type is None:
    loss_fn = nn.CrossEntropyLoss()
    loss_val = loss_fn(output, targets)
else:
    loss_fn = nn.MSELoss()
    loss_val = torch.sqrt(loss_fn(output, codes) + 1e-6)

# printing the accuracy of the network without training
print_batch_accuracy(data, targets, train=True)

# clear previously stored gradients
optimizer.zero_grad()

# calculate the gradients
loss_val.backward()

# learning rate update
optimizer.step()

# calculate new network outputs using the same data
output = net(data.view(batch_size, -1))
if code_type is None:
    loss_val = loss_fn(output, targets)
else:
    loss_val = torch.sqrt(loss_fn(output, codes) + 1e-6)

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

    # Fault synapse train (gradual situation)
    if FS == "train" and FG:
        fault_mask = torch.zeros_like(torch.zeros(layer_size)).to(device)
        if FT == "sporadic":
            size = int(len(fault_next_pos) * (epoch - FE) / (num_epochs - FE - 1))
            for i in range(size):
                for j in fault_prev_pos[i]:
                    fault_mask[j, fault_next_pos[i]] = stuck

        elif FT == "linear":
            size = int(len(fault_next_pos) * (epoch - FE) / (num_epochs - FE - 1))
            for i in fault_prev_pos:
                for j in range(size):
                    fault_mask[i, fault_next_pos[j]] = stuck

        elif FT == "arial":
            size = int(len(fault_next_pos) * (epoch - FE) / (num_epochs - FE - 1))
            for i in range(size):
                for j in range(size):
                    fault_mask[fault_prev_pos[i], fault_next_pos[j]] = stuck

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)
        if code_type is None:
            codes = None
        else:
            codes = code_convert(targets)

        # forward pass
        net.train()
        output = net(data.view(batch_size, -1))

        # initialize the loss & sum over time
        if code_type is None:
            loss_val = loss_fn(output, targets)
        else:
            loss_val = torch.sqrt(loss_fn(output, codes) + 1e-6)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():

            # Fault synapse train
            if epoch > FE and FS == "train":
                list(net.parameters())[(FL - 1) * 2].data *= torch.where(fault_mask.transpose(0, 1) != 0, 0,
                                                                         torch.ones_like(
                                                                             fault_mask.transpose(0, 1)))
                if mod:
                    fault_mask = fault_mask.bool() * (torch.randn_like(fault_mask) * std + stuck)

                list(net.parameters())[(FL - 1) * 2].data += fault_mask.transpose(0, 1)

            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)
            if code_type is None:
                test_codes = None
            else:
                test_codes = code_convert(test_targets)

            # Test set forward pass
            test_output = net(test_data.view(batch_size, -1))

            # Test set loss
            if code_type is None:
                test_loss = loss_fn(test_output, test_targets)
            else:
                test_loss = torch.sqrt(loss_fn(test_output, test_codes) + 1e-6)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer()
            counter += 1
            iter_counter += 1

n_sqrth = int(np.ceil(np.sqrt(1024)))
h = get_square_weights(list(net.parameters())[0].view(784, 1024), n_sqrth, 28)
o = list(net.parameters())[2].view(512, 1024)

total = 0
correct = 0

# drop_last switched to False value to keep all samples
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)

w_targets = torch.tensor([], dtype=dtype).to(device)
w_predicted = torch.tensor([], dtype=dtype).to(device)

with torch.no_grad():
    net.eval()

    # Fault synapse test
    if FS == "test":
        list(net.parameters())[(FL - 1) * 2].data *= torch.where(fault_mask.transpose(0, 1) != 0, 0,
                                                                 torch.ones_like(fault_mask.transpose(0, 1)))
        if mod:
            fault_mask = fault_mask.bool() * (torch.randn_like(fault_mask) * std + stuck)

        list(net.parameters())[(FL - 1) * 2].data += fault_mask.transpose(0, 1)

    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        test_output = net(data.view(data.size(0), -1))

        # calculate total accuracy
        if code_type == "bin":
            test_pred = bin_distance_pred(test_output, batch_size)
        elif code_type == "mul":
            test_pred = mul_distance_pred(test_output, batch_size)
        else:
            test_pred = test_output.argmax(dim=1)
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

    heat_ih = plt.figure(facecolor="w", figsize=(8, 8))
    plt.imshow(h.detach().cpu().numpy(), cmap="hot", interpolation="nearest")
    plt.title("Layer 1 to Layer 2 Weights")
    plt.colorbar()
    plt.show()

    heat_ho = plt.figure(facecolor="w", figsize=(8, 8))
    plt.imshow(o.detach().cpu().numpy(), cmap="hot", interpolation="nearest")
    plt.title("Layer 2 to Layer 3 Weights")
    plt.colorbar()
    plt.show()

    cm_fig = plt.figure(figsize=(8, 8))
    cm = confusion_matrix(w_targets, w_predicted)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    plt.title("Confusion Matrix")
    plt.show()
