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
x = data[:, :-1]
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

print("Running on Device =", device)
beta = 0.9  # neuron decay rate
spike_grad = surrogate.fast_sigmoid()

#  Initialize Network
num_hidden1 = 64
num_hidden2 = 32

net = nn.Sequential(nn.Linear(num_inputs, num_hidden1),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Linear(num_hidden1, num_hidden2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Linear(num_hidden2, num_outputs),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
                    ).to(device)


def forward_pass(net, data, num_steps):
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)


def test_accuracy(data_loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

    data_loader = iter(data_loader)

    for data, targets in data_loader:
        data = data.to(device)
        targets = targets.to(device)
        spk_rec = forward_pass(net, data, num_steps)

        acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
        total += spk_rec.size(1)

    return acc/total

optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

loss_hist = []
acc_hist = []

# Initialize empty lists to store training and testing accuracy
acc_table = []
test_accs = []

ST = False
AST = False
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

drop_num = 10
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

# training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec = forward_pass(net, data, num_steps)
        loss_val = loss_fn(spk_rec, targets)

        # Synaptic Template
        if ST:
            for k in range(len(template_exc)):
                for j in drop_input[k]:
                    list(net.parameters())[0].data[template_exc[k], j] = 0
                for j in reinforce_input[k]:
                    if list(net.parameters())[0].data[template_exc[k], j] <= 0.5:
                        list(net.parameters())[0].data[template_exc[k], j] = \
                            reinforce_ref[int(template_exc[k])][int(np.where(j == reinforce_input[k])[0])] * 0.5

        # Dead synapses
        if DS:
            for i in range(len(dead_out)):
                for j in dead_input[i]:
                    list(net.parameters())[2].data[dead_out[i], j] = 0

        # Synaptic Weight Scaling
        if SC:
            list(net.parameters())[2].data *= 1.0001  # 1.0001

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

            for k in range(num_hidden2):
                for j in range(num_hidden1):
                    m[k, j] = int(bernoulli.rvs(p[k, j], size=1))

            list(net.parameters())[2].data *= m

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # print every 25 iterations
        if i % iteration == 0:
            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            # check accuracy on a single batch
            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")
            acc_table.append((epoch, i, acc))

    # Calculate testing accuracy
    test_acc = test_accuracy(test_loader, net, num_steps)
    test_accs.append(test_acc)

    print(f"Epoch {epoch} \nTest Accuracy: {test_acc * 100:.3f}%\n")

acc_df = pd.DataFrame(acc_table, columns=['Epoch', 'Iteration', 'Accuracy'])

if plot == True:
    plt.plot(acc_hist)
    plt.xlabel(f"Iterations (every {iteration} batches)")
    plt.ylabel('Accuracy')

    # Create the figure and axis objects for the plot
    fig, ax = plt.subplots()
    test_line, = ax.plot(test_accs, label='Testing Accuracy')

    ax.set_title("Testing Accuracy over Time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    # Add the legend to the plot
    ax.legend(handles=[test_line])
    plt.show()
