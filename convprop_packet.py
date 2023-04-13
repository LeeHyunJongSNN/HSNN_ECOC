import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import torch
import random
import torch.nn as nn

import snntorch as snn
from snntorch import utils
import snntorch.functional as SF
from snntorch import surrogate
from snntorch import backprop
from snntorch import spikeplot as splt

from IPython.display import HTML
from sklearn.preprocessing import minmax_scale
from scipy.stats import bernoulli

import matplotlib.pyplot as plt

batch_size = 128
num_steps = 25
iteration = 100
train_rate = 0.8
num_epochs = 2
label_style = "binary"
plot = True
data_path = "concat_all_minmax.csv"

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
    print("Error: data_path is not defined.")
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
data = torch.tensor(frame.values, dtype=torch.float64).float()

# Split the data into features (x) and labels (y)
x = data[:, :-1].reshape(-1, 4, 19).unsqueeze(1)
y = data[:, -1].long()

# Convert the features and labels into a PyTorch dataset
dataset = torch.utils.data.TensorDataset(x, y)

# Split the dataset into a training set and a test set
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(train_rate * len(dataset)), len(dataset) - int(train_rate * len(dataset))])

# Create loader using the dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Start
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# neuron and simulation parameters
print("Running on Device =", device)
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 50

#  Initialize Network
net = nn.Sequential(nn.Conv2d(1, 12, kernel_size=5, stride=1, padding=2),
                    nn.AdaptiveMaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 64, kernel_size=5, stride=1, padding=2),
                    nn.AdaptiveMaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64 * 4, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

for step in range(num_steps):
    spk_out, mem_out = net(data)

def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)

spk_rec, mem_rec = forward_pass(net, num_steps, data)

loss_fn = SF.ce_rate_loss()
loss_val = loss_fn(spk_rec, targets)
acc = SF.accuracy_rate(spk_rec, targets)
print(f"The loss from an untrained network is {loss_val.item():.3f}")
print(f"The accuracy of a single batch using an untrained network is {acc * 100:.3f}%")

def batch_accuracy(train_loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

    train_loader = iter(train_loader)

    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)
        spk_rec, _ = forward_pass(net, num_steps, data)

        acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
        total += spk_rec.size(1)

    return acc/total

test_acc = batch_accuracy(test_loader, net, num_steps)
print(f"The total accuracy on the test set is: {test_acc * 100:.2f}%")

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 10
test_acc_hist = []

# training loop
for epoch in range(num_epochs):

    avg_loss = backprop.BPTT(net, train_loader, optimizer=optimizer, criterion=loss_fn,
                             num_steps=num_steps, time_var=False, device=device)

    print(f"Epoch {epoch}, Train Loss: {avg_loss.item():.2f}")

    # Test set accuracy
    test_acc = batch_accuracy(test_loader, net, num_steps)
    test_acc_hist.append(test_acc)

    print(f"Epoch {epoch}, Test Acc: {test_acc * 100:.2f}%\n")

# Plot Loss
fig = plt.figure(facecolor="w")
plt.plot(test_acc_hist)
plt.title("Test Set Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

spk_rec, mem_rec = forward_pass(net, num_steps, data)

idx = 0

fig2, ax = plt.subplots(facecolor='w', figsize=(12, 7))
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(f"The target label is: {targets[idx]}")

#  Plot spike count histogram
anim = splt.spike_count(spk_rec[:, idx].detach().cpu(), fig2, ax, labels=labels,
                        animate=True, interpolate=4)

HTML(anim.to_html5_video())
