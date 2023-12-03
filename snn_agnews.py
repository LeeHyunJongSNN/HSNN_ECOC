import snntorch as snn
from snntorch import surrogate

import torch
import torch.nn as nn
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import argparse
import seaborn as sns

from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS

dtype = torch.float

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--num_steps", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--ECOC", type=bool, default=False)
parser.add_argument("--FS", type=str, default="train")
parser.add_argument("--FT", type=str, default="sporadic")
parser.add_argument("--FG", type=bool, default=False)
parser.add_argument("--FE", type=int, default=4)
parser.add_argument("--FL", type=int, default=2)
parser.add_argument("--FR", type=float, default=0.1)
parser.add_argument("--stuck", type=float, default=0.1)
parser.add_argument("--mod", type=bool, default=False)
parser.add_argument("--std", type=float, default=0.5)
parser.add_argument("--gpu_num", type=int, default=0)
parser.add_argument("--plot", type=bool, default=False)

args = parser.parse_args()

batch_size = args.batch_size
num_steps = args.num_steps
num_epochs = args.num_epochs
ECOC = args.ECOC
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

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


train_iter, test_iter = AG_NEWS()
train_set = to_map_style_dataset(train_iter)
test_set = to_map_style_dataset(test_iter)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
vocab_size = len(vocab)

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
)

test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
)

# Define codes for ECOC
if ECOC:
    defined_code = torch.tensor([[1, 0, 1, 0, 0, 1],
                                 [1, 1, 0, 0, 1, 0],
                                 [1, 0, 0, 1, 0, 0],
                                 [1, 1, 1, 1, 1, 1]], dtype=dtype).to(device)
    num_classes = len(defined_code)
    output_num = len(defined_code[0])

else:
    output_num = len(set([label for (label, text) in train_iter]))

# Fault synapse simulation
if FL == 1:
    layer_size = (128, 64)
elif FL == 2:
    layer_size = (64, 32)
elif FL == 3:
    layer_size = (32, output_num)
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
        fault_mask = torch.zeros_like(torch.zeros(layer_size)).to(device)
        fault_prev_pos = random.sample(range(0, layer_size[0]), FS_prev_num)
        fault_next_pos = random.sample(range(0, layer_size[1]), FS_next_num)
        for i in fault_prev_pos:
            for j in fault_next_pos:
                fault_mask[i, j] = stuck

    elif FT == "arial":
        fault_mask = torch.zeros_like(torch.zeros(layer_size)).to(device)
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
spike_grad = surrogate.fast_sigmoid(slope=25)
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.embedding = nn.EmbeddingBag(len(vocab), 128, sparse=False)
        self.fc1 = nn.Linear(128, 64)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad)
        self.fc2 = nn.Linear(64, 32)
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad)
        self.fc3 = nn.Linear(32, output_num)
        self.lif3 = snn.Leaky(beta=0.95, spike_grad=spike_grad, output=True)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()

    def forward(self, x, offsets):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record the final layer
        spk3_rec = []
        mem3_rec = []

        for step in range(num_steps):
            embedded = self.embedding(x, offsets)
            cur1 = self.fc1(embedded)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

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

def distance_pred(output: torch.Tensor, batch_size: int):
    bin_output = output.sum(dim=0).bool().int()
    pred = torch.zeros(batch_size, num_classes).to(device)
    for i in range(batch_size):
        for j in range(num_classes):
            ham_dist = distance.hamming(bin_output[i].detach().cpu().numpy(), defined_code[j].detach().cpu().numpy())
            pred[i, j] = ham_dist

    return pred.argmin(dim=1)

def print_batch_accuracy(data, offsets, targets, train=False):
    out_spk, _ = net(data, offsets)
    if ECOC:
        train_pred = distance_pred(out_spk, batch_size)
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
    print_batch_accuracy(data, offsets, targets, train=True)
    print_batch_accuracy(test_data, test_offsets, test_targets, train=False)
    print("\n")

optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
sample_batch = enumerate(train_loader)
idx, (targets, data, offsets) = next(sample_batch)
data = data.to(device)
targets = targets.to(device)
offsets = offsets.to(device)
if ECOC:
    codes = code_convert(targets)
else:
    codes = None

spk_rec, mem_rec = net(data, offsets)

# initialize the loss function
loss_val = torch.zeros(1, dtype=dtype, device=device)
loss_fn = nn.CrossEntropyLoss()
for step in range(num_steps):
    if ECOC:
        loss_val += loss_fn(mem_rec[step], codes)
    else:
        loss_val += loss_fn(mem_rec[step], targets)

# printing the accuracy of the network without training
print_batch_accuracy(data, offsets, targets, train=True)

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
print_batch_accuracy(data, offsets, targets, train=True)

loss_hist = []
test_loss_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = enumerate(train_loader)
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
    for idx, (targets, data, offsets) in train_batch:
        data = data.to(device)
        targets = targets.to(device)
        offsets = offsets.to(device)
        if ECOC:
            codes = code_convert(targets)
        else:
            codes = None

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data, offsets)

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

            # Fault synapse train
            if epoch > FE and FS == "train":
                list(net.parameters())[(FL - 1) * 2].data *= torch.where(fault_mask.transpose(0, 1) != 0, 0,
                                                                         torch.ones_like(
                                                                             fault_mask.transpose(0, 1)))
                if mod:
                    fault_mask = fault_mask.bool() * (torch.randn_like(fault_mask) * std + stuck)

                list(net.parameters())[(FL - 1) * 2].data += fault_mask.transpose(0, 1)

            net.eval()
            idx, (test_targets, test_data, test_offsets) = next(enumerate(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)
            test_offsets = test_offsets.to(device)
            if ECOC:
                test_codes = code_convert(test_targets)
            else:
                test_codes = None

            # Test set forward pass
            test_spk, test_mem = net(test_data, test_offsets)

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

n_sqrth = int(np.ceil(np.sqrt(1024)))
h = list(net.parameters())[1].view(128, 64)
o = list(net.parameters())[3].view(64, 32)

total = 0
correct = 0

w_targets = torch.tensor([], dtype=dtype).to(device)
w_predicted = torch.tensor([], dtype=dtype).to(device)

with torch.no_grad():
    net.eval()

    # Fault synapse test
    if FS == "test":
        list(net.parameters())[FL * 2 - 1].data *= torch.where(fault_mask.transpose(0, 1) != 0, 0,
                                                               torch.ones_like(fault_mask.transpose(0, 1)))
        if mod:
            fault_mask = fault_mask.bool() * (torch.randn_like(fault_mask) * std + stuck)

        list(net.parameters())[FL * 2 - 1].data += fault_mask.transpose(0, 1)

    for idx, (targets, data, offsets) in enumerate(test_loader):
        data = data.to(device)
        targets = targets.to(device)
        offsets = offsets.to(device)

        # forward pass
        test_spk, test_mem = net(data, offsets)

        # calculate total accuracy
        if ECOC:
            test_pred = distance_pred(test_spk, batch_size)
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
