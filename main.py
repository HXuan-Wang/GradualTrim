import copy
import gc
import torch.nn as nn
from prune_funcs import prune_DNN_neck_module,prune_DNN_bottom_module
import torch_pruning as tp
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from data import imagenet
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models import *
from torch.utils.data import DataLoader,Subset,RandomSampler

import os
import argparse

from data import imagenet

from utils import progress_bar

import utils

import torchvision.models

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./data',
    help='dataset path')
parser.add_argument(
    '--train_data_dir',
    type=str,
    default='./data',
    help='dataset path')

parser.add_argument(
    '--dataset',
    type=str,
    default='imagenet_fewshot',
    choices=('synthetic_imagenet', 'imagenet', 'imagenet_fewshot'),
    help='dataset')
parser.add_argument(
    '--lr',
    default=0.01,
    type=float,
    help='initial learning rate')
parser.add_argument(
    '--lr_decay_step',
    default='5,10',
    type=str,
    help='learning rate decay step')
parser.add_argument(
    '--resume',
    type=str,
    default=None,
    help='load the model from the specified checkpoint')
parser.add_argument('--num_sample', type=int, default=50,
                    help='number of samples for training')
parser.add_argument('--seed', type=int, default=0,
                    help='seed')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')
parser.add_argument(
    '--job_dir',
    type=str,
    default='./result/tmp/',
    help='The directory where the summaries will be stored.')

parser.add_argument(
    '--epochs',
    type=int,
    default=15,
    help='The num of epochs to train.')
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=128,
    help='Batch size for training.')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=64,
    help='Batch size for validation.')

parser.add_argument(
    '--arch',
    type=str,
    default='resnet50',
    choices=('resnet50', 'resnet34', 'vgg16'),
    help='The architecture to prune')

parser.add_argument('--norm_type', type=str, default='mean')
parser.add_argument('--local_imptype', type=str, default='mag')
parser.add_argument('--global_imptype', type=str, default='mag')
parser.add_argument('--local_prune_ratio', type=float, default=0.5)
parser.add_argument('--global_prune_ratio', type=float, default=0.5)

args = parser.parse_args()


def test(epoch, net, optimizer=None, scheduler=None, testloader=None):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    global best_acc

    net.eval()
    num_iterations = len(testloader)
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

        print_logger.info(
            'Epoch[{0}]({1}/{2}): '
            'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                epoch, batch_idx, num_iterations, top1=top1, top5=top5))

    if top1.avg > best_acc:
        print_logger.info('Saving to ' + args.arch + '.pt')
        state = {
            'state_dict': net.state_dict(),
            'best_prec1': top1.avg,
            'epoch': epoch,
            # 'scheduler':scheduler.state_dict(),
            # 'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir(args.job_dir + '/pruned_checkpoint'):
            os.mkdir(args.job_dir + '/pruned_checkpoint')
        best_acc = top1.avg
        torch.save(state, args.job_dir + '/pruned_checkpoint/' + args.arch +  '.pt')

    print_logger.info("=>Best accuracy {:.3f}".format(best_acc))

def Local_distillation(train_loader, test_loader, model, model_t, args, device=None, feature_layer_list=[]):

    criterion = torch.nn.MSELoss(reduction='mean').cuda()

    optimizer_step1 = optim.SGD(
        model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4) # 0.0005

    feature_epoch = args.epochs

    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_step1, eta_min=1e-5, T_max=feature_epoch)
    model.eval()
    test(0, model, optimizer_step1, scheduler1, test_loader)

    model.train()
    model_t.eval()

    iter_nums = 0

    torch.cuda.empty_cache()
    correct = 0
    total = 0

    if args.dataset == 'synthetic_imagenet':
        for i in range(feature_epoch):
            for batch_idx, (inputs, targets) in enumerate(train_loader):

                inputs = inputs.cuda()
                targets = targets.cuda()
                optimizer_step1.zero_grad()

                outputs, s_features = model(inputs, return_layer_feature=True, feature_layer_list=feature_layer_list)
                with torch.no_grad():
                    t_output, t_features = model_t(inputs, return_layer_feature=True, feature_layer_list=feature_layer_list)
                loss_list = []

                for i in range(len(s_features)):
                    loss_i = criterion(s_features[i], t_features[i])
                    loss_list.append(loss_i)

                loss = torch.sum(torch.stack(loss_list))
                loss.backward()


                optimizer_step1.step()

                train_loss = loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(trainloader
                                            ),
                             ' Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss, 100. * correct / total, correct, total))

                scheduler1.step()

        test(iter_nums, model, optimizer_step1, scheduler1, test_loader)

    else:
        finish = False
        while not finish:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                iter_nums += 1

                if iter_nums > feature_epoch:
                    finish = True
                    break
                inputs = inputs.cuda()
                targets = targets.cuda()
                optimizer_step1.zero_grad()


                outputs, s_features = model(inputs, return_layer_feature=True, feature_layer_list=feature_layer_list)
                with torch.no_grad():
                    t_output, t_features = model_t(inputs, return_layer_feature=True, feature_layer_list=feature_layer_list)

                loss_list = []

                for i in range(len(s_features)):
                    loss_i = criterion(s_features[i], t_features[i])
                    loss_list.append(loss_i)

                loss =  torch.sum(torch.stack(loss_list))
                loss.backward()

                optimizer_step1.step()

                train_loss = loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_idx, len(trainloader
                                            ),
                             ' Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss, 100. * correct / total, correct, total))

                if iter_nums % args.eval_freq == 0:
                    test(iter_nums,model, optimizer_step1, scheduler1, test_loader)
                    model.train()

                scheduler1.step()

def Global_distillation(train_loader, test_loader, model, model_t, args, device=None):



    criterion = torch.nn.MSELoss(reduction='mean').cuda()
    #
    optimizer_step1 = optim.SGD(
        model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4) # 0.0005

    feature_epoch = args.epochs

    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_step1, eta_min=0, T_max=feature_epoch)
    model.eval()
    test(0, model, optimizer_step1, scheduler1, test_loader)


    # switch to train mode
    model.train()

    model_t.eval()

    iter_nums = 0


    torch.cuda.empty_cache()
    correct = 0
    total = 0


    if args.dataset == 'synthetic_imagenet':
        for i in range(feature_epoch):
            for batch_idx, (inputs, targets) in enumerate(train_loader):

                inputs = inputs.cuda()
                targets = targets.cuda()
                optimizer_step1.zero_grad()

                outputs, s_features = model(inputs, return_last_feature=True)
                with torch.no_grad():
                    t_output, t_features = model_t(inputs, return_last_feature=True)

                loss = criterion(s_features, t_features)

                loss.backward()
                optimizer_step1.step()

                train_loss = loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(trainloader
                                            ),
                             ' Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss, 100. * correct / total, correct, total))

                scheduler1.step()

        test(iter_nums, model, optimizer_step1, scheduler1, test_loader)

    else:
        finish = False
        while not finish:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                iter_nums += 1

                if iter_nums > feature_epoch:
                    finish = True
                    break
                inputs = inputs.cuda()
                targets = targets.cuda()
                optimizer_step1.zero_grad()
                outputs, s_features = model(inputs,return_last_feature=True)
                with torch.no_grad():
                    t_output, t_features = model_t(inputs,return_last_feature=True)

                loss = criterion(s_features, t_features)
                loss.backward()

                optimizer_step1.step()

                train_loss = loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(trainloader
                                            ),
                             ' Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss, 100. * correct / total, correct, total))

                if iter_nums % args.eval_freq == 0:
                    test(iter_nums, model, optimizer_step1, scheduler1, test_loader)
                    model.train()


                scheduler1.step()

def find_conv_layers(model):
    conv_layers=[]
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            conv_layers.append((name))
    return conv_layers

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if len(args.gpu) == 1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
lr_decay_step = list(map(int, args.lr_decay_step.split(',')))

ckpt = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
utils.print_params(vars(args), print_logger.info)

# Data


print_logger.info('==> Preparing data..')

if args.dataset == 'imagenet':
    data_tmp = imagenet.Data(args)
    trainloader = data_tmp.loader_train
    testloader = data_tmp.loader_test

elif args.dataset == 'synthetic_imagenet':
    traindir = args.train_data_dir
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scale_size = 224
    trainset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(scale_size),
            transforms.ToTensor(),
            normalize,
        ]))

    trainloader = DataLoader(
        trainset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=False)
    data_tmp = imagenet.Data(args)
    testloader = data_tmp.loader_test

elif args.dataset == 'imagenet_fewshot':
    args.eval_freq = 1000
    trainloader = imagenet.imagenet_fewshot(img_num=args.num_sample, batch_size=min(args.train_batch_size,args.num_sample), seed=args.seed,
                                            data_dir=args.data_dir)

    data_tmp = imagenet.Data(args)

    testloader = data_tmp.loader_test

    try:
        trainloader.dataset.samples_to_file(os.path.join(args.save, "samples.txt"))
    except:
        print('Not save samples.txt')




# Model
model_name = args.arch
print_logger.info('==> Building model..')


model_s = build_model(model_name)
model_t = build_model(model_name)

feature_layer_list=[]

if model_name=='vgg16':
    conv_layers = find_conv_layers(model_s)
    conv_layers = conv_layers[1:-1]
    indices = [int(layer.split('.')[-1]) for layer in conv_layers]
    feature_layer_list = indices[1::2]
    print("feature_layer_list before pruning",feature_layer_list)

#[7, 14, 20, 27, 34]

model_t = model_t.to(device)
model_s = model_s.to(device)
local_ratio = args.local_prune_ratio
global_ratio = args.global_prune_ratio
norm_type = args.norm_type
local_imptype = args.local_imptype
global_imptype = args.global_imptype

cudnn.benchmark = True

criterion = nn.CrossEntropyLoss().cuda()

example_inputs = torch.randn(1, 3, 224, 224)

print("===========================Pruning Start===========================")

print("stage1")
print(model_s)
model_s.cpu().eval()

model_s = prune_DNN_neck_module(model=model_s, example_inputs=example_inputs,
                            model_name=model_name, round_to=None, ratio=local_ratio, imptype=local_imptype,
                            norm_type=norm_type,global_way=True,eval_loader=trainloader,device=device) #0.4
model_s.to(device)

# print(model_s)



Local_distillation(trainloader, testloader, model_s, model_t, args, device=device,feature_layer_list=feature_layer_list)



model_s.cpu().eval()
model_s = prune_DNN_bottom_module(model=model_s, example_inputs=example_inputs,
                                      model_name=model_name, round_to=None, ratio=global_ratio, imptype=global_imptype,
                                      norm_type=norm_type, global_way=True,eval_loader=trainloader,device=device) #0.3
model_s.to(device)
# print("pruned models after stage2")
# print(model_s)

Global_distillation(trainloader, testloader, model_s, model_t, args, device=device)