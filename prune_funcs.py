import numpy as np
import torch
import torch.nn as nn
import random
# from segment_anything_kd import SamPredictor, sam_model_registry
# from segment_anything_kd.modeling.image_encoder import Attention
from statistics import mean
from torch.nn.functional import threshold, normalize
# from segment_anything_kd.modeling.common import LayerNorm2d
import torch.nn.functional as F
import torch_pruning as tp
import pickle
import os
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def find_conv_layers(model):
    conv_layers=[]
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            conv_layers.append((name))
    return conv_layers

def prune_DNN_neck_module(model, example_inputs, model_name, round_to, ratio, imptype, norm_type, global_way,eval_loader,device):
    ignored_layers = []

    #########################################
    # Ignore unprunable modules
    #########################################

    # ignored_layers.append(model.fc)

    if model_name=='resnet34':
        ignored_layers.append(model.conv1)
        ignored_layers.append(model.fc)
        block_index = [3,4,6,3]
        for i in range(len(block_index)):
            block = eval(('model.layer%d' % (i + 1)))
            for j in range(block_index[i]):
                ignored_layers.append(block[j].conv2)
                # ignored_layers.append(block[j].conv1)
                # ignored_layers.append(block[j].conv3)
        block = eval(('model.layer%d' % (3 + 1)))
        ignored_layers.append(block[2].conv2)
    elif model_name=='vgg16':

        ignored_layers.append(model.classifier)
        conv_layers=find_conv_layers(model)

        ignored_layers.append(model.features[int(conv_layers[0].split('.')[-1])])
        ignored_layers.append(model.features[int(conv_layers[-1].split('.')[-1])])
        conv_layers=conv_layers[1:-1]
        indices=[int(layer.split('.')[-1]) for layer in conv_layers]
        local_mask_list=indices[1::2]
        print(local_mask_list)
        for i in range(len(local_mask_list)):
            ignored_layers.append(model.features[int(local_mask_list[i])])




    round_to = None

    #########################################
    # (Optional) Register unwrapped nn.Parameters
    # TP will automatically detect unwrapped parameters and prune the last dim for you by default.
    # If you want to prune other dims, you can register them here.
    #########################################
    unwrapped_parameters = None

    #########################################
    # Build network pruners
    #########################################

    if imptype == "Disturb":
        importance = tp.importance.DisturbImportance(normalizer=norm_type, group_reduction="mean")
    elif imptype == "mag":
        importance = tp.importance.MagnitudeImportance(p=2, normalizer=norm_type)
    elif imptype == "taylor":
        importance = tp.importance.TaylorImportance(normalizer=norm_type, group_reduction="mean")
    elif imptype == "random":
        importance = tp.importance.RandomImportance()
    elif imptype == "Local_Disturb":
        importance = tp.importance.Local_DisturbImportance(normalizer=norm_type, group_reduction="mean")
    elif imptype == "GLobal_Disturb":
        importance = tp.importance.Global_DisturbImportance(normalizer=norm_type, group_reduction="mean")




    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=iterative_steps,
        pruning_ratio=ratio,

        round_to=round_to,
        unwrapped_parameters=unwrapped_parameters,
        ignored_layers=ignored_layers,
        global_pruning=global_way,
    )

    #########################################
    # Pruning
    #########################################

    for i in range(iterative_steps):

        ori_macs, ori_size = tp.utils.count_ops_and_params(model, example_inputs)

        if imptype == "Local_Disturb" or imptype == 'GLobal_Disturb':
            model = model.to(device)
            model.zero_grad()
            imp = pruner.importance
            imp._prepare_model(model, pruner)

            for k, (imgs, lbls) in enumerate(eval_loader):
                if k >= 5: #resnet_34:100
                    break
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                output = model(imgs).to(device)

                loss = F.cross_entropy(output, lbls).to(device)
                loss.backward()

            pruner.step()
            imp._rm_hooks(model)
            m = imp._clear_buffer()

            model = model.cpu()
        else:
            pruner.step()
        #########################################
        # Testing
        #########################################

        with torch.no_grad():
            if isinstance(example_inputs, dict):
                out = model(**example_inputs)
            else:
                out = model(example_inputs)
            print("{} Intra block Pruning:: ".format(model_name))
            macs_after_prune, params_after_prune = tp.utils.count_ops_and_params(model, example_inputs)
            print(" Params: %s => %s" % (ori_size, params_after_prune))
            print(" Macs: %s => %s" % (ori_macs, macs_after_prune))
            print('Mac pruning percentage:{}'.format((ori_macs- macs_after_prune) / ori_macs * 100))
            print('Params pruning percentage:{}'.format((ori_size - params_after_prune) / ori_size * 100))


            if isinstance(out, (dict, list, tuple)):
                print("  Output:")
                for o in tp.utils.flatten_as_list(out):
                    print(o.shape)
            else:
                print("  Output:", out.shape)
            print("------------------------------------------------------\n")

    return model



def prune_DNN_bottom_module(model, example_inputs, model_name, round_to, ratio, imptype, norm_type, global_way,eval_loader,device):
    ignored_layers = []

    #########################################
    # Ignore unprunable modules
    #########################################

    # ignored_layers.append(model.fc)
    if model_name == 'resnet34':

        ignored_layers.append(model.conv1)
        ignored_layers.append(model.fc)
        block_index = [3,4,6,3]
        for i in range(len(block_index)):
            block = eval(('model.layer%d' % (i + 1)))
            for j in range(block_index[i]):
                ignored_layers.append(block[j].conv1) #conv2 within pruning conv1 across_block_pruning
                # ignored_layers.append(block[j].conv2)

        block = eval(('model.layer%d' % (3 + 1)))
        ignored_layers.append(block[2].conv2)
    elif model_name == 'vgg16':
        ignored_layers.append(model.classifier)
        conv_layers = find_conv_layers(model)
        ignored_layers.append(model.features[int(conv_layers[0].split('.')[-1])])
        ignored_layers.append(model.features[int(conv_layers[-1].split('.')[-1])])
        conv_layers = conv_layers[1:-1]
        indices = [int(layer.split('.')[-1]) for layer in conv_layers]
        global_mask_list = indices[::2]
        print(global_mask_list)
        for i in range(len(global_mask_list)):
            ignored_layers.append(model.features[global_mask_list[i]])
    round_to = None

    #########################################
    # (Optional) Register unwrapped nn.Parameters
    # TP will automatically detect unwrapped parameters and prune the last dim for you by default.
    # If you want to prune other dims, you can register them here.
    #########################################
    unwrapped_parameters = None

    #########################################
    # Build network pruners
    #########################################

    if imptype == "Disturb":
        importance = tp.importance.DisturbImportance(normalizer=norm_type, group_reduction="mean")
    elif imptype == "mag":
        importance = tp.importance.MagnitudeImportance(p=2, normalizer=norm_type)
    elif imptype == "taylor":
        importance = tp.importance.TaylorImportance(normalizer=norm_type, group_reduction="mean")
    elif imptype == "random":
        importance = tp.importance.RandomImportance()
    elif imptype == "Local_Disturb":
        importance = tp.importance.Local_DisturbImportance(normalizer=norm_type, group_reduction="mean")
    elif imptype == "GLobal_Disturb":
        importance = tp.importance.Global_DisturbImportance(normalizer=norm_type, group_reduction="mean")




    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=iterative_steps,
        pruning_ratio=ratio,
        # pruning_ratio_dict={
        #     model.layer1: 0.2,
        #     model.layer2: 0.3,
        #     model.layer3: 0.4,
        #     model.layer4: 0.4
        # },
        round_to=round_to,
        unwrapped_parameters=unwrapped_parameters,
        ignored_layers=ignored_layers,
        global_pruning=global_way,
    )

    #########################################
    # Pruning
    #########################################

    for i in range(iterative_steps):

        ori_macs, ori_size = tp.utils.count_ops_and_params(model, example_inputs)

        if imptype == "Local_Disturb" or imptype == 'GLobal_Disturb':
            model = model.to(device)
            model.zero_grad()
            imp = pruner.importance
            imp._prepare_model(model, pruner)

            for k, (imgs, lbls) in enumerate(eval_loader):
                if k >= 10:
                    break
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                output = model(imgs).to(device)

                loss = F.cross_entropy(output, lbls).to(device)
                loss.backward()

            pruner.step()
            imp._rm_hooks(model)
            imp._clear_buffer()
            model = model.cpu()
        else:
            pruner.step()
        #########################################
        # Testing
        #########################################



        with torch.no_grad():
            if isinstance(example_inputs, dict):
                out = model(**example_inputs)
            else:
                out = model(example_inputs)
            print("{} Inter block Pruning: ".format(model_name))
            macs_after_prune, params_after_prune = tp.utils.count_ops_and_params(model, example_inputs)
            print(" Params: %s => %s" % (ori_size/1e6, params_after_prune/1e6))
            print(" Macs: %s => %s" % (ori_macs/1e9, macs_after_prune/1e9))
            print('Mac pruning percentage:{}'.format((ori_macs- macs_after_prune) / ori_macs * 100))
            print('Params pruning percentage:{}'.format((ori_size - params_after_prune) / ori_size * 100))


            if isinstance(out, (dict, list, tuple)):
                print("  Output:")
                for o in tp.utils.flatten_as_list(out):
                    print(o.shape)
            else:
                print("  Output:", out.shape)
            print("------------------------------------------------------\n")

    return model