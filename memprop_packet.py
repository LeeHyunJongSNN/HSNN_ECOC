import snntorch as snn
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import minmax_scale
from scipy.stats import bernoulli

import matplotlib.pyplot as plt
import numpy as np
import random

# dataloader arguments
batch_size = 64
train_ratio = 0.8
label_style = "binary"
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_path = "concat_all_minmax_PCA.csv"

if data_path == "concat_all_minmax_PCA.csv":
    Feature = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10', 'Label']
    num_inputs = 10

elif data_path == "concat_all_minmax.csv":
    Feature = [' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 'Total Length of Fwd Packets',
               ' Total Length of Bwd Packets',
               ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
               'Bwd Packet Length Max',
               ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow IAT Mean',
               ' Flow IAT Std',
               ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
               ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
               ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags',
               ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
               ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
               ' Packet Length Variance',
               'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count',
               ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
               ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
               ' Fwd Avg Packets/Bulk',
               ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
               'Subflow Fwd Packets',
               ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
               ' Init_Win_bytes_backward',
               ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max',
               ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min',
               ' Inbound', 'Label']
    num_inputs = 76

else:
    print("Error: data_path is not defined")
    exit()

# Define a Dataset
df = pd.read_csv(data_path, encoding='cp949')
df = df[Feature]  # Select Feature

if label_style == "multiple":
    df.replace({'Label': 'normal'}, 0, inplace=True)
    df.replace({'Label': 'Syn'}, 1, inplace=True)
    df.replace({'Label': 'DrDoS_DNS'}, 2, inplace=True)
    df.replace({'Label': 'DrDoS_LDAP'}, 3, inplace=True)
    df.replace({'Label': 'DrDoS_MSSQL'}, 4, inplace=True)
    df.replace({'Label': 'DrDoS_NTP'}, 5, inplace=True)
    df.replace({'Label': 'DrDoS_UDP'}, 6, inplace=True)
    df.replace({'Label': 'TFTP'}, 7, inplace=True)
    df.replace({'Label': 'UDP-lag'}, 8, inplace=True)
    df.replace({'Label': 'DrDoS_NetBIOS'}, 9, inplace=True)
    df.replace({'Label': 'DrDoS_SNMP'}, 10, inplace=True)
    df.replace({'Label': 'DrDoS_SSDP'}, 11, inplace=True)
    df.replace({'Label': 'Portmap'}, 12, inplace=True)
    num_outputs = 13

elif label_style == "binary":
    df.replace({'Label': 'normal'}, 0, inplace=True)
    df.replace({'Label': 'Syn'}, 1, inplace=True)
    df.replace({'Label': 'DrDoS_DNS'}, 1, inplace=True)
    df.replace({'Label': 'DrDoS_LDAP'}, 1, inplace=True)
    df.replace({'Label': 'DrDoS_MSSQL'}, 1, inplace=True)
    df.replace({'Label': 'DrDoS_NTP'}, 1, inplace=True)
    df.replace({'Label': 'DrDoS_UDP'}, 1, inplace=True)
    df.replace({'Label': 'TFTP'}, 1, inplace=True)
    df.replace({'Label': 'UDP-lag'}, 1, inplace=True)
    df.replace({'Label': 'DrDoS_NetBIOS'}, 1, inplace=True)
    df.replace({'Label': 'DrDoS_SNMP'}, 1, inplace=True)
    df.replace({'Label': 'DrDoS_SSDP'}, 1, inplace=True)
    df.replace({'Label': 'Portmap'}, 1, inplace=True)
    num_outputs = 2

frame = df.sample(frac=1, random_state=0)
all_Num = len(frame)

# Convert the frame DataFrame into a PyTorch tensor
raw_data = torch.tensor(frame.values, dtype=torch.float64).float()

# Split the data into features (x) and labels (y)
x = raw_data[:, :-1]
y = raw_data[:, -1].long()

# Convert the features and labels into a PyTorch dataset
dataset = torch.utils.data.TensorDataset(x, y)

# Split the dataset into a training set and a test set
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(train_ratio * len(dataset)), len(dataset) - int(train_ratio * len(dataset))])

# Create loader using the dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Network Architecture
num_hidden1 = 64
num_hidden2 = 64
num_hidden3 = 64

# Temporal Dynamics
num_steps = 25
beta = 0.95
# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden1)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(num_hidden2, num_hidden3)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc4 = nn.Linear(num_hidden3, num_outputs)
        self.lif4 = snn.Leaky(beta=beta)


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
AST = True
ADC = False
DS = False
SC = False

print("ST: ", ST)
if ST:
    print("AST: ", AST)
print("ADC: ", ADC)
print("DS: ", DS)
print("SC: ", SC)

DS_out_num = 1
DS_hidden_num = 64

drop_num = 1
reinforce_num = 1
pre_average = []
dropout_index = []
drop_input = []
reinforce_input = []
reinforce_ref = []
dead_input = []

preprocessed = x.numpy()
pre_size = 51000
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

drop_input *= int(np.ceil(num_hidden1 / 10))
reinforce_input *= int(np.ceil(num_hidden1 / 10))
reinforce_ref *= int(np.ceil(num_hidden1 / 10))
template_exc = np.arange(num_hidden1)

dead_input = []
if DS:
    for i in range(DS_out_num):
        dead_input.append(random.sample(range(0, num_outputs), DS_hidden_num))
dead_out = random.sample(range(0, num_outputs), DS_out_num)

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
                    list(net.parameters())[0].data[template_exc[i], j] = 0
                for j in reinforce_input[i]:
                    if list(net.parameters())[0].data[template_exc[i], j] <= 0.5:
                        list(net.parameters())[0].data[template_exc[i], j] = \
                            reinforce_ref[int(template_exc[i])][int(np.where(j == reinforce_input[i])[0])] * 0.5

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
                np.nan_to_num(list(net.parameters())[2].data.cpu().detach().numpy().reshape(num_hidden2 * num_hidden1),
                              copy=False),
                feature_range=(0.9995, 1)).reshape(num_hidden2, num_hidden1), 3)
            m = torch.zeros(num_hidden2, num_hidden1).to('cuda')

            for i in range(num_hidden2):
                for j in range(num_hidden1):
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
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

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
