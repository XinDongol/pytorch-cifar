'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
# from utils import progress_bar

from pprint import pprint
import math
from torch.nn.init import _calculate_fan_in_and_fan_out, calculate_gain


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG16')


def new_kaiming_uniform_(tensor, a=0, nonlinearity='leaky_relu'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = 0.5*(fan_in + fan_out)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def new_kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = 0.5*(fan_in + fan_out)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)

def new_init_(tensor, c=2/3):
    kernel_size = tensor.size(2)
    var = c/(kernel_size * math.sqrt(tensor.size(0)*tensor.size(1)))
    with torch.no_grad():
        return tensor.normal_(0, math.sqrt(var))

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0, nonlinearity='relu')
        # new_kaiming_normal_(m.weight)
        new_init_(m.weight)

# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

pprint(net)
input('press any to continue ...')


class FeatureHook():
    def __init__(self, name, module):
        self.name = name
        self.module = module
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        
        self.inp = input[0]
        if self.inp.requires_grad:
            self.inp.retain_grad()
        self.out = output
        if self.out.requires_grad:
            self.out.retain_grad()

        # r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
        #     module.running_mean.data.type(var.type()) - mean, 2)

        # self.r_feature = r_feature

    # def hook_fn_backward(self, module, inp_grad, out_grad):
    #     self.inp_grad = inp_grad
    #     self.out_grad = out_grad

    def close(self):
        self.hook.remove()

hooks = {}

for n,m in net.named_modules():
    if isinstance(m, nn.Conv2d):
        hooks[n] = FeatureHook(n, m)


global_step = 0
# Training
def train(epoch):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        global_step += 1
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if global_step == 3 or global_step == 5000:
            # print('-'*20+'grad abs mean'+'-'*20)
            # for n, h in hooks.items():
            #     print(n, str(h.module)[:16], ' | ', '%.3e'%h.module.weight.grad.abs().mean().item())

            print('-'*20+'grad std'+'-'*20)
            for n, h in hooks.items():
                print(n, str(h.module)[:16], ' | ', '%.3e'%h.module.weight.grad.std().item())  

            print('-'*20+'grad std / weight std'+'-'*20)
            for n, h in hooks.items():
                print(n, str(h.module)[:16], ' | ', '%.3e'%(h.module.weight.grad.std().item()/h.module.weight.std().item()))  

            # print('-'*20+'inp grad abs mean'+'-'*20)
            # for n, h in hooks.items():
            #     if h.inp.grad is not None:
            #         print(n, str(h.module)[:16], ' | ', '%.3e'%h.inp.grad.abs().mean().item())

            print('-'*20+'inp grad std'+'-'*20)
            for n, h in hooks.items():
                if h.inp.grad is not None:
                    print(n, str(h.module)[:16], ' | ', '%.3e'%h.inp.grad.std().item())  

            # print('-'*20+'out grad abs mean'+'-'*20)
            # for n, h in hooks.items():
            #     print(n, str(h.module)[:16], ' | ', '%.3e'%h.out.grad.abs().mean().item())

            print('-'*20+'out grad std'+'-'*20)
            for n, h in hooks.items():
                print(n, str(h.module)[:16], ' | ', '%.3e'%h.out.grad.std().item())  

            # print('-'*20+'inp mean'+'-'*20)
            # for n, h in hooks.items():
            #     print(n, str(h.module)[:16], ' | ', '%.3e'%h.inp.mean().item())

            print('-'*20+'inp std'+'-'*20)
            for n, h in hooks.items():
                print(n, str(h.module)[:16], ' | ', '%.3e'%h.inp.std().item())          

            # print('-'*20+'out mean'+'-'*20)
            # for n, h in hooks.items():
            #     print(n, str(h.module)[:16], ' | ', '%.3e'%h.out.mean().item())

            print('-'*20+'out std'+'-'*20)
            for n, h in hooks.items():
                print(n, str(h.module)[:16], ' | ', '%.3e'%h.out.std().item())

            print('-'*20+'weight abs mean'+'-'*20)
            for n, h in hooks.items():
                print(n, str(h.module)[:16], ' | ', '%.3e'%h.module.weight.abs().mean().item())

            print('##'*20)
            print()

            if global_step == 5000:
                assert 1==0


        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(batch_idx, epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(batch_idx, epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
