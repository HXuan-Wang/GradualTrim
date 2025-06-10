import abc
import torch
import torch.nn as nn

import typing
from collections import OrderedDict
from ..utils.compute_mat_grad import ComputeMatGrad
from . import function
from ..dependency import Group
from .._helpers import _FlattenIndexMapping
from .. import ops
import math
from ..utils import *
import numpy as np
import warnings

import abc
import warnings
import torch
import torch.nn as nn
from sklearn.linear_model import LassoLars
import time
import copy


__all__ = [
    "Importance",
    "MagnitudeImportance",
    "BNScaleImportance",
    "LAMPImportance",
    "RandomImportance",
    "TaylorImportance",
    "HessianImportance",
    
    # Group Importance
    "GroupNormImportance",
    "GroupTaylorImportance",
    "GroupHessianImportance",
]

class Importance(abc.ABC):
    """ Estimate the importance of a tp.Dependency.Group, and return an 1-D per-channel importance score.

        It should accept a group and a ch_groups as inputs, and return a 1-D tensor with the same length as the number of channels.
        ch_groups refer to the number of internal groups, e.g., for a 64-channel **group conv** with groups=ch_groups=4, each group has 16 channels.
        All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
        Just ignore the ch_groups if you are not familar with grouping.

        Example:
            ```python
            DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
            group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
            scorer = MagnitudeImportance()    
            imp_score = scorer(group, ch_groups=1)    
            #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
            min_score = imp_score.min() 
            ``` 
    """
    @abc.abstractclassmethod
    def __call__(self, group: Group, ch_groups: int=1) -> torch.Tensor: 
        raise NotImplementedError


class MagnitudeImportance(Importance):
    """ A general implementation of magnitude importance. By default, it calculates the group L2-norm for each channel/dim.
        MagnitudeImportance supports several variants:
            - Standard L1-norm for single layer: MagnitudeImportance(p=1, normalizer=None, group_reduction="first")
            - Group L1-Norm: MagnitudeImportance(p=1, normalizer=None, group_reduction="mean")
            - BN Scaling Factor: MagnitudeImportance(p=1, normalizer=None, group_reduction="mean", target_types=[nn.modules.batchnorm._BatchNorm])

        Args:
            * p (int): the norm degree. Default: 2
            * group_reduction (str): the reduction method for group importance. Default: "mean"
            * normalizer (str): the normalization method for group importance. Default: "mean"
            * target_types (list): the target types for importance calculation. Default: [nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm]
    """
    def __init__(self, 
                 p: int=2, 
                 group_reduction: str="mean", 
                 normalizer: str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.LSTM]):
        self.p = p
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias

    def _lamp(self, scores): # Layer-adaptive Sparsity for the Magnitude-based Pruning
        """
        Normalizing scheme for LAMP.
        """
        # sort scores in an ascending order
        sorted_scores,sorted_idx = scores.view(-1).sort(descending=False)
        # compute cumulative sum
        scores_cumsum_temp = sorted_scores.cumsum(dim=0)
        scores_cumsum = torch.zeros(scores_cumsum_temp.shape,device=scores.device)
        scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
        # normalize by cumulative sum
        sorted_scores /= (scores.sum() - scores_cumsum)
        # tidy up and output
        new_scores = torch.zeros(scores_cumsum.shape,device=scores.device)
        new_scores[sorted_idx] = sorted_scores
        
        return new_scores.view(scores.shape)

    # def _lamp(self, imp): # Layer-adaptive Sparsity for the Magnitude-based Pruning
    #     argsort_idx = torch.argsort(imp, dim=0, descending=True).tolist()
    #     sorted_imp = imp[argsort_idx]
    #     cumsum_imp = torch.cumsum(sorted_imp, dim=0)
    #     sorted_imp = sorted_imp / cumsum_imp
    #     inversed_idx = torch.arange(len(sorted_imp))[
    #         argsort_idx
    #     ].tolist()  # [0, 1, 2, 3, ..., ]
    #     return sorted_imp[inversed_idx]
    
    def _normalize(self, group_importance, normalizer):
        if normalizer is None:
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (group_importance.max() - group_importance.min()+1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std()+1e-8)
        elif normalizer.startswith('sentinel'): # normalize the score with the k-th smallest element. e.g. sentinel_0.5 means median normalization
            sentinel = float(normalizer.split('_')[1]) * len(group_importance)
            sentinel = torch.argsort(group_importance, dim=0, descending=False)[int(sentinel)]
            return group_importance / (group_importance[sentinel]+1e-8)
        elif normalizer=='lamp':
            return self._lamp(group_importance)
        else:
            raise NotImplementedError

    def _reduce(self, group_imp: typing.List[torch.Tensor], group_idxs: typing.List[typing.List[int]]):
        if len(group_imp) == 0: return group_imp
        if self.group_reduction == 'prod':
            reduced_imp = torch.ones_like(group_imp[0])
        elif self.group_reduction == 'max':
            reduced_imp = torch.ones_like(group_imp[0]) * -99999
        else:
            reduced_imp = torch.zeros_like(group_imp[0])

        for i, (imp, root_idxs) in enumerate(zip(group_imp, group_idxs)):
            if self.group_reduction == "sum" or self.group_reduction == "mean":
                reduced_imp.scatter_add_(0, torch.tensor(root_idxs, device=imp.device), imp) # accumulated importance
            elif self.group_reduction == "max": # keep the max importance
                selected_imp = torch.index_select(reduced_imp, 0, torch.tensor(root_idxs, device=imp.device))
                selected_imp = torch.maximum(input=selected_imp, other=imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), selected_imp)
            elif self.group_reduction == "prod": # product of importance
                selected_imp = torch.index_select(reduced_imp, 0, torch.tensor(root_idxs, device=imp.device))
                torch.mul(selected_imp, imp, out=selected_imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), selected_imp)
            elif self.group_reduction == 'first':
                if i == 0:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), imp)
            elif self.group_reduction == 'gate':
                if i == len(group_imp)-1:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), imp)
            elif self.group_reduction is None:
                reduced_imp = torch.stack(group_imp, dim=0) # no reduction
            else:
                raise NotImplementedError
        
        if self.group_reduction == "mean":
            reduced_imp /= len(group_imp)
        return reduced_imp
    
    @torch.no_grad()
    def __call__(self, group: Group, ch_groups: int=1):
         # 1. 首先定义一个列表用于存储分组内每一层的重要性
        group_imp = []
        group_idxs = []
        # 2. 迭代分组内的各个层，对Conv层计算重要性
        # Iterate over all groups and estimate group importance
        # print("group: ", group)
        for i, (dep, idxs) in enumerate(group):  # idxs是一个包含所有可剪枝索引的列表，用于处理DenseNet中的局部耦合的情况
            layer = dep.layer  # 获取 nn.Module
            # print("layer: ", layer)
            prune_fn = dep.pruning_fn  # 获取 剪枝函数
            # print("pruning_fn: ", prune_fn)
            root_idxs = group[i].root_idxs
            # if not isinstance(layer, tuple(self.target_types)):
            #     print("layer: ", layer, " 没有匹配的target_types!!")
            #     continue
            # else:
            #     print("layer: ", layer, " 有匹配的target_types~~")
            #     print("它的prune_fn是:", prune_fn)
            ####################
            # Conv/Linear Output
            ####################
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                # print("线性或卷积输出层", layer)
                # print("layer.weight shape", layer.weight.shape)
                # print("检测到输出通道")
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    # print("type(layer.weight.data): ", type(layer.weight.data))
                    w = layer.weight.data[idxs].flatten(1)  # 用索引列表获取耦合通道对应的参数，并展开成2维
                local_imp = w.abs().pow(self.p).sum(1)   # 计算每个通道参数子矩阵的 self.p Norm
                # print("layer: ", layer)
                # print("w shape", w.shape)
                # print("local_imp shape", local_imp.shape)
                # print("local_imp", local_imp)
                # print("local_imp shape", local_imp.shape)
                group_imp.append(local_imp)  # 将其保存在列表中
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    local_imp = layer.bias.data[idxs].abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            ####################
            # Conv/Linear Input
            ####################
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                # print("线性或卷积输入层", layer)
                # print("layer.weight shape", layer.weight.shape)
                # print("检测到输入通道")
                if prune_fn == function.prune_conv_in_channels and layer.groups == layer.in_channels:  # 如果分组卷积组数等于通道数则跳过
                    continue
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight.data).flatten(1)
                else:
                    w = (layer.weight.data).transpose(0, 1).flatten(1)
                local_imp = w.abs().pow(self.p).sum(1)
                # print("layer: ", layer)
                # print("w shape", w.shape)
                # print("local_imp shape", local_imp.shape)
                # print("local_imp", local_imp)

                # if local_imp.numel() == 1:
                #     print("group", group)

                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    # print("检测到分组卷积层: ", layer)
                    local_imp = local_imp.repeat(layer.groups)
                    # print("处理后的 local_imp shape", local_imp.shape)
                    # print("处理后的 local_imp", local_imp)
                
                local_imp = local_imp[idxs]
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            ####################
            # LSTM Output
            ####################
            if prune_fn in [
                function.prune_lstm_out_channels,
            ]:
                # print("lstm out channels weight_hh_l0 w shape: ", layer.weight_hh_l0.data.shape)
                # print("idxs len: ", len(idxs))
                # print("idxs: ", idxs)
                # print("(layer.weight_hh_l0.data)[idxs]", (layer.weight_hh_l0.data)[idxs].shape)
                w = layer.weight_hh_l0.transpose(0, 1).data
                # print("w shape: ", w.shape)
                local_imp = w.abs().pow(self.p).sum(1)
                # print("local_imp shape: ", local_imp.shape)
                group_imp.append(local_imp)  # 将其保存在列表中
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    local_imp = layer.bias_hh_l0.data[idxs].abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                # w = (layer.weight_ih_l0.data).flatten(1)

                # print("lstm out channels weight_ih_l0 w shape: ", w.shape)
                # local_imp = w.abs().pow(self.p).sum(1)
                # group_imp.append(local_imp)  # 将其保存在列表中
                # group_idxs.append(root_idxs)

                # if self.bias and layer.bias is not None:
                #     local_imp = layer.bias_ih_l0.data[idxs].abs().pow(self.p)
                #     group_imp.append(local_imp)
                #     group_idxs.append(root_idxs)

            ####################
            # LSTM Input
            ####################
            elif prune_fn in [
                function.prune_lstm_in_channels,
            ]:
                # w = (layer.weight_hh_l0.data)[idxs].flatten(1)
                # print("lstm in channels weight_ih_l0 w shape: ", layer.weight_ih_l0.data.shape)
                # print("idxs len: ", len(idxs))
                # print("idxs: ", idxs)
                # print("(layer.weight_ih_l0.data).transpose(0, 1)[idxs]", (layer.weight_ih_l0.data).transpose(0, 1)[idxs].shape)
                # local_imp = w.abs().pow(self.p).sum(1)
                # local_imp = local_imp[idxs]
                # group_imp.append(local_imp)
                # group_idxs.append(root_idxs)
                w = (layer.weight_ih_l0.data).transpose(0, 1)
                # print("w shape: ", w.shape)
                # print("lstm in channels weight_ih_l0 w shape: ", w.shape)
                local_imp = w.abs().pow(self.p).sum(1)
                local_imp = local_imp[idxs]
                # print("local_imp shape: ", local_imp.shape)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            ####################
            # BatchNorm
            ####################
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                # print("BN层", layer)
                if layer.affine:
                    # print("layer.weight shape", layer.weight.shape)
                    w = layer.weight.data[idxs]
                    # print("w shape", w.shape)
                    local_imp = w.abs().pow(self.p)
                    # print("local_imp shape", local_imp.shape)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs().pow(self.p)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            ####################
            # LayerNorm
            ####################
            elif prune_fn == function.prune_layernorm_out_channels:

                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    local_imp = w.abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs().pow(self.p)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


class BNScaleImportance(MagnitudeImportance):
    """Learning Efficient Convolutional Networks through Network Slimming, 
    https://arxiv.org/abs/1708.06519
    """

    def __init__(self, group_reduction='mean', normalizer='mean'):
        super().__init__(p=1, group_reduction=group_reduction, normalizer=normalizer, bias=False, target_types=(nn.modules.batchnorm._BatchNorm,))


class LAMPImportance(MagnitudeImportance):
    """Layer-adaptive Sparsity for the Magnitude-based Pruning,
    https://arxiv.org/abs/2010.07611
    """

    def __init__(self, p=2, group_reduction="mean", normalizer='lamp', bias=False):
        assert normalizer == 'lamp'
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer, bias=bias)

class FPGMImportance(MagnitudeImportance):
    """Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration,
    http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.pdf
    """

    def __init__(self, p=2, group_reduction="mean", normalizer='mean', bias=False):
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer, bias=bias)

    @torch.no_grad()
    def __call__(self, group, **kwargs):
        group_imp = []
        group_idxs = []
        # Iterate over all groups and estimate group importance
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue
            ####################
            # Conv/Linear Output
            ####################
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                    
                
                local_imp = w.abs().pow(self.p)
                
                # calculate the euclidean distance as similarity
                similar_matrix = torch.cdist(local_imp.unsqueeze(0), local_imp.unsqueeze(0), p=2).squeeze(0)
                similar_sum = torch.sum(torch.abs(similar_matrix), dim=0)
                # print("similar_sum is",similar_sum.shape)
                group_imp.append(similar_sum)
                group_idxs.append(root_idxs)

            ####################
            # Conv/Linear Input
            ####################
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight.data).flatten(1)
                else:
                    w = (layer.weight.data).transpose(0, 1).flatten(1)

                local_imp = w.abs().pow(self.p)

                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(ch_groups)
                local_imp = local_imp[idxs]
                similar_matrix = torch.cdist(local_imp.unsqueeze(0), local_imp.unsqueeze(0), p=2).squeeze(0)
                similar_sum = torch.sum(torch.abs(similar_matrix), dim=0)
                group_imp.append(similar_sum)
                group_idxs.append(root_idxs)

            # FPGMImportance should not care about BatchNorm and LayerNorm

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class RandomImportance(Importance):
    @torch.no_grad()
    def __call__(self, group, **kwargs):
        _, idxs = group[0]
        return torch.rand(len(idxs))


class TaylorImportance(MagnitudeImportance):
    """First-order taylor expansion of the loss function.
       https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 multivariable:bool=False, 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.multivariable = multivariable
        self.target_types = target_types
        self.bias = bias

    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            # print("layer is",layer)
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue
            
            # Conv/Linear Output
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                    dw = layer.weight.grad.data.transpose(1, 0)[
                        idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                    dw = layer.weight.grad.data[idxs].flatten(1)
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    b = layer.bias.data[idxs]
                    db = layer.bias.grad.data[idxs]
                    local_imp = (b * db).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    
            # Conv/Linear Input
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight).flatten(1)
                    dw = (layer.weight.grad).flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)
                    dw = (layer.weight.grad).transpose(0, 1).flatten(1)
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                
                # print("local_imp is",local_imp)
                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                local_imp = local_imp[idxs]
                

                group_imp.append(local_imp)
                group_idxs.append(root_idxs)


            # BN
            elif prune_fn == function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w*dw).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            
            # LN
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w*dw).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp
    

class OBDCImportance(MagnitudeImportance):
    """EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis:
       http://proceedings.mlr.press/v97/wang19g/wang19g.pdf
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear],
                 num_classes=100):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self.A, self.DS = {}, {}
        self.Fisher = {}
        self.MatGradHandler = ComputeMatGrad()
        self.steps = 0
        self.eps = 1e-10
        self.modules = []
        self.num_classes = num_classes
        self.known_modules = {'Linear', 'Conv2d'}
    
    def step(self):
        with torch.no_grad():
            for m in self.modules:
                A, DS = self.A[m], self.DS[m]
                # print("m",m)
                # print("A.shape",A.shape)
                # print("DS.shape",DS.shape)
                # if A is None:
                #     continue
                grad_mat = self.MatGradHandler(A, DS, m)
                grad_mat *= DS.size(0)
                if self.steps == 0:
                    self.Fisher[m] = grad_mat.new(grad_mat.size()[1:]).fill_(0)
                self.Fisher[m] += (grad_mat.pow_(2)).sum(0)
                # print("self.Fisher[m].shape: ", self.Fisher[m].shape)
                # print("m.weight.shape: ", m.weight.shape)
                # self.A[m] = None
                # self.DS[m] = None
        self.steps += 1

    def adjust_fisher(self, group, idxs):
        for i, (dep, id) in enumerate(group):
            layer = dep.target.module
            if layer in self.modules:
                if layer.weight.grad is not None:
                    shape = layer.weight.shape
                    if isinstance(layer, nn.modules.conv._ConvNd):
                        kernel_size = shape[2]*shape[3]
                    else:
                        kernel_size = 1
                    indices_to_keep = list(range(self.Fisher[layer].shape[1]))
                    for idx in idxs:
                        indices_to_keep = [i for i in indices_to_keep if not (idx*kernel_size <= i < (idx+1)*kernel_size)]
                    self.Fisher[layer] = torch.index_select(self.Fisher[layer], 1, torch.tensor(indices_to_keep).to(self.Fisher[layer].device))
            

    def _rm_hooks(self, model):
        for m in self.modules:
            m._backward_hooks = OrderedDict()
            m._forward_pre_hooks = OrderedDict()

    def _save_input(self, module, input):
        self.A[module] = input[0].data

    def _save_grad_output(self, module, grad_input, grad_output):
        self.DS[module] = grad_output[0].data

    def _prepare_model(self, model, pruner):
        for group in pruner.DG.get_all_groups(ignored_layers=pruner.ignored_layers, root_module_types=pruner.root_module_types): 
            group = pruner._downstream_node_as_root_if_attention(group)
            for i, (dep, idxs) in enumerate(group):
                layer = dep.target.module
                if isinstance(layer, tuple(self.target_types)) and dep.handler in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    self.modules.append(layer)
                    layer.register_forward_pre_hook(self._save_input)
                    layer.register_backward_hook(self._save_grad_output)

    def _clear_buffer(self):
        self.Fisher = {}
        self.modules = []
        self.steps = 0
    
    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)) or (isinstance(layer, torch.nn.Linear) and layer.out_features == self.num_classes):
                continue
            F_diag = (self.Fisher[layer] / self.steps + self.eps)
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                    else:
                        w = layer.weight.data[idxs].flatten(1)
                    # print("w.shape: ", w.shape)
                    # print("F_diag.shape: ", F_diag.shape)
                    local_imp = (w ** 2 * F_diag).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                
                if self.bias and layer.bias is not None and layer.bias.grad is not None:
                    b = layer.bias.data[idxs]
                    local_imp = (b ** 2 * F_diag).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class CPImportance(MagnitudeImportance):
    """EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis:
       http://proceedings.mlr.press/v97/wang19g/wang19g.pdf
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear],
                num_classes=100):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self.num_classes = num_classes
        self.Input = {}
        self.modules = []

    def _rm_hooks(self, model):
        for m in self.modules:
            m._forward_pre_hooks = OrderedDict()

    def _save_input(self, module, input):
        # print("module: ", module)
        weight_shape = module.weight.shape
        if isinstance(module, torch.nn.Linear):
            self.Input[module] = input[0]
        else:
            B, c_in, H, W = input[0].shape
            # print("B: ", B)
            # print("c_in: ", c_in)
            # print("H: ", H)
            # print("W: ", W)
            out = torch.zeros((B, c_in, weight_shape[2], weight_shape[3])).to(input[0].device)

            for i in range(B):
                for j in range(c_in):
                    start_height = torch.randint(0, H - weight_shape[2] + 1, (1,)).item()
                    start_width = torch.randint(0, W - weight_shape[3] + 1, (1,)).item()
                    patch = input[0][i, j, start_height:start_height + weight_shape[2], start_width:start_width + weight_shape[3]]
                    out[i, j] = patch
            self.Input[module] = out

    def _prepare_model(self, model, pruner):
        for group in pruner.DG.get_all_groups(ignored_layers=pruner.ignored_layers, root_module_types=pruner.root_module_types): 
            group = pruner._downstream_node_as_root_if_attention(group)
            for i, (dep, idxs) in enumerate(group):
                layer = dep.target.module
                if isinstance(layer, tuple(self.target_types)) and dep.handler in [
                    function.prune_conv_in_channels,
                    function.prune_linear_in_channels,
                ]:
                    self.modules.append(layer)
                    layer.register_forward_pre_hook(self._save_input)


    def _clear_buffer(self):
        self.Input = {}
        self.modules = []
    
    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
        group_idxs = []
        # print("group: ", group)
        for i, (dep, idxs) in enumerate(group):
            layer = dep.target.module
            prune_fn = dep.handler
            # print("layer: ", layer)
            # print("prune_fn: ", prune_fn)
            # print("prune_fn is",prune_fn)
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)) or (isinstance(layer, torch.nn.Linear) and layer.out_features == self.num_classes):
                continue
            # print("layer1  is ",layer)
            # print("prune_fn is",prune_fn)   
            if prune_fn in[
                function.prune_conv_in_channels,
                function.prune_linear_in_channels
            ]:
                X = self.Input[layer]
                W = layer.weight
                # print("W is",W.shape)
                # print("X.shape before: ", X.shape)
                if len(W.shape) == 4:
                    Y = torch.nn.functional.conv2d(X, W, padding=0).squeeze(-1).squeeze(-1)
                else:
                    Y = torch.mean(torch.matmul(X, W.T), dim=1)  # [B, c_out]
                    X = torch.mean(X, dim=1) # [B, c_in]
                # print("Y.shape: ", Y.shape)
                num_samples = self.Input[layer].shape[0]  # num of samples
                c_in = W.shape[1]  # num of input channels
                c_out = W.shape[0]  # num of output channels
                # print("layer is",layer)
                # print("X.shape: ", X.shape)
                # print("layer is",layer)
                # print("X.shape: ", X.shape)
                # print("W.shape: ", W.shape)
                # conv
                if len(W.shape) == 4:
                    # sample and reshape X to [c_in, B, 9]
                    reshape_X = X.reshape((num_samples, c_in, -1)).permute(1, 0, 2)
                    # reshape W to [c_in, 9, c_out]
                    reshape_W = W.reshape((c_out, c_in, -1)).permute(1, 2, 0)
                # linear
                else:
                    # X's shape is [B, token_lenn c_in]
                    reshape_X = X.reshape(c_in, -1, 1)
                    # reshape W to [c_in, 1, c_out]
                    reshape_W =  W.reshape((c_out, c_in, 1)).permute(1, 2, 0)
                    # # sample and reshape X to [c_in, B] and expand to [c_in, B, 1]
                    # reshape_X = X.transpose((1, 0))[...,np.newaxis]
                    # # reshape to [c_in, 1, c_out]
                    # reshape_W =  W.reshape((c_out, c_in, 1)).permute(1, 2, 0)
                # print("reshape_X.shape: ", reshape_X.shape)
                # print("reshape_W.shape: ", reshape_W.shape)
                # reshape Y to [B x c_out]
                reshape_Y = Y.reshape(-1).cpu()
                # print("reshape_Y.shape: ", reshape_Y.shape)
                # product has size [B x c_out, c_in]
                product = np.matmul(reshape_X.cpu(), reshape_W.cpu()).reshape((c_in, -1)).T
                # print("product.shape: ", product.shape)
                # use LassoLars because it's more robust than Lassos
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    solver = LassoLars(alpha=0, fit_intercept=False)  # , max_iter=3000
                    solver.fit(product, reshape_Y)

                # print("solver.coef_[idxs]",solver.coef_[idxs].shape)
                local_imp = np.abs(solver.coef_[idxs])
                # print("local_imp is",local_imp.shape)


                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                # local_imp = local_imp[idxs]
                group_imp.append(torch.from_numpy(local_imp))
                group_idxs.append(root_idxs)

        # print("layer1 is",layer)
        # print("group imp is",group_imp)
        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)

        
        return group_imp

class HRankImportance(MagnitudeImportance):
    """Hrank: Filter pruning using high-rank feature map:
       https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_HRank_Filter_Pruning_Using_High-Rank_Feature_Map_CVPR_2020_paper.pdf
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear],
                num_classes=100):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self.num_classes = num_classes
        self.feature_result = {}
        self.total = {}
        self.modules = []

    def _rm_hooks(self, model):
        for m in self.modules:
            m._forward_hooks = OrderedDict()

    #get feature map of certain layer via hook
    def get_feature_hook(self, module, input, output):
        if module not in self.feature_result.keys():
            self.feature_result[module] = torch.tensor(0.).to(output.device)
        if module not in self.total.keys():
            self.total[module] = torch.tensor(0.).to(output.device)
        num_features = output.shape[1]
        batch_norm = nn.BatchNorm2d(num_features=num_features).to(output.device)
        res_feature = batch_norm(output)
        relu = nn.ReLU(inplace=False)
        res_feature = relu(res_feature)
        a = output.shape[0]
        b = output.shape[1]
        c = torch.tensor([torch.linalg.matrix_rank(res_feature[i,j,:,:]) for i in range(a) for j in range(b)]).to(output.device)
        c = c.view(a, -1).float()
        c = c.sum(0)
        self.feature_result[module] = self.feature_result[module] * self.total[module] + c
        self.total[module] = self.total[module] + a
        self.feature_result[module] = self.feature_result[module] / self.total[module]

    def _prepare_model(self, model, pruner):
        for group in pruner.DG.get_all_groups(ignored_layers=pruner.ignored_layers, root_module_types=pruner.root_module_types): 
            group = pruner._downstream_node_as_root_if_attention(group)
            for i, (dep, idxs) in enumerate(group):
                layer = dep.target.module
                if isinstance(layer, tuple(self.target_types)) and dep.handler in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    self.modules.append(layer)
                    layer.register_forward_hook(self.get_feature_hook)


    def _clear_buffer(self):
        self.feature_result = {}
        self.total = {}
        self.modules = []
    
    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)) or (isinstance(layer, torch.nn.Linear) and layer.out_features == self.num_classes):
                continue
            if prune_fn in[
                function.prune_conv_out_channels,
                function.prune_linear_out_channels
            ]:
                group_imp.append(self.feature_result[layer])
                group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp





class ThiNetImportance(MagnitudeImportance):
    """EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis:
       http://proceedings.mlr.press/v97/wang19g/wang19g.pdf
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear],
                num_classes=100):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self.num_classes = num_classes
        self.Input = {}
        self.modules = []

    def _rm_hooks(self, model):
        for m in self.modules:
            m._forward_pre_hooks = OrderedDict()


    def _save_input(self, module, input):  # 这个为啥不删
        # print("module: ", module)
        # weight_shape = module.weight.shape
        self.Input[module] = input[0] # # Input是单个元素（特征图）的元组
        # print("input[0] shape: ", input[0].shape)

        # 无用的计算
        # if isinstance(module, torch.nn.Linear):
        #     self.Input[module] = input[0]
        # else:
        #     B, c_in, H, W = input[0].shape
        #     # print("B: ", B)
        #     # print("c_in: ", c_in)
        #     # print("H: ", H)
        #     # print("W: ", W)
        #     out = torch.zeros((B, c_in, weight_shape[2], weight_shape[3])).to(input[0].device)

        #     # 啊？
        #     # z这两个方法都是用的这个作为输入吗
        #     # 这是在一个特征图上取一个3x3的小块，ThiNet也是这样？
        #     for i in range(B):
        #         for j in range(c_in):
        #             start_height = torch.randint(0, H - weight_shape[2] + 1, (1,)).item()
        #             start_width = torch.randint(0, W - weight_shape[3] + 1, (1,)).item()
        #             patch = input[0][i, j, start_height:start_height + weight_shape[2], start_width:start_width + weight_shape[3]]
        #             out[i, j] = patch
        #     self.Input[module] = out

    def _prepare_model(self, model, pruner):
        for group in pruner.DG.get_all_groups(ignored_layers=pruner.ignored_layers, root_module_types=pruner.root_module_types): 
            group = pruner._downstream_node_as_root_if_attention(group)
            for i, (dep, idxs) in enumerate(group):
                layer = dep.target.module
                if isinstance(layer, tuple(self.target_types)) and dep.handler in [
                    function.prune_conv_in_channels,
                    function.prune_linear_in_channels,
                ]:
                    self.modules.append(layer)
                    layer.register_forward_pre_hook(self._save_input)


    def _clear_buffer(self):
        self.Input = {}
        self.modules = []
    
    @torch.no_grad()
    def __call__(self, group,ch_groups=1):
        group_imp = []
        group_idxs = []
        indices_pruned = []
        min_diff = 1e10
        min_idx = 0
        
        for i, (dep, idxs) in enumerate(group):
            local_imp = []
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)) or (isinstance(layer, torch.nn.Linear) and layer.out_features == self.num_classes):
                continue
            if prune_fn in[
                function.prune_conv_in_channels,
                function.prune_linear_in_channels
            ]:
                X = self.Input[layer]
                # print("X shape: ", X.shape)  # [32, 512, 3, 3] --> [32, 512, 4, 4]
                W = layer.weight
                # # print("W is",W.shape)
                # # print("X.shape before: ", self.Input[layer].shape)

                # # 这些都是无用计算  你只需要X
                # if len(W.shape) == 4:
                #     Y = torch.nn.functional.conv2d(X, W, padding=0).squeeze(-1).squeeze(-1)
                # else:
                #     Y = torch.mean(torch.matmul(X, W.T), dim=1)  # [B, c_out]
                #     X = torch.mean(X, dim=1)# [B, c_in]


                # # print("Y.shape: ", Y.shape)
                # num_samples = self.Input[layer].shape[0]  # num of samples
                c_in = W.shape[1]  # num of input channels
                # c_out = W.shape[0]  # num of output channels

                # print("layer is",layer)
                # print("X.shape: ", X.shape)
                # print("W.shape: ", W.shape)

                # print("c_in shape is",c_in)
                # conv
                # 这是你加的噶
                # 就加了这里噶
                # 这就是唯一的计算
                for idx in range(c_in):
                    if idx in indices_pruned:
                        continue
                    indices_try = indices_pruned + [idx]
                    inputs_try = torch.zeros_like(X)
                    if len(W.shape) == 4:  # 卷积层
                        inputs_try[:, indices_try, ...] = X[:, indices_try, ...]
                    else:  # 线性层
                        inputs_try[:, :, indices_try] = X[:, :, indices_try]
                    output_try = layer(inputs_try)
                    output_try_norm = output_try.norm(2)
                    # print("output_try_norm shape is",output_try_norm.shape)
                    local_imp.append(output_try_norm.item())
                    # print("local_imp shape is",len(local_imp))
                    # print("local_imp is",local_imp)
                    # local_imp = torch.tensor(local_imp)
                    # print("local_imp shape is",local_imp.shape)   
                # print("local_imp  is",local_imp)  
                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                
                local_imp_tensor = torch.tensor(copy.deepcopy(local_imp))
                group_imp.append(local_imp_tensor)
                group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        
        return group_imp





    
class HessianImportance(MagnitudeImportance):
    """Optimal Brain Damage:
       https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self._accu_grad = {}
        self._counter = {}

    def zero_grad(self):
        self._accu_grad = {}
        self._counter = {}

    def accumulate_grad(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self._accu_grad:
                    self._accu_grad[param] = param.grad.data.clone().pow(2)
                else:
                    self._accu_grad[param] += param.grad.data.clone().pow(2)
                
                if name not in self._counter:
                    self._counter[param] = 1
                else:
                    self._counter[param] += 1
    
    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
        group_idxs = []

        if len(self._accu_grad) > 0: # fill gradients so that we can re-use the implementation for Taylor
            for p, g in self._accu_grad.items():
                p.grad.data = g / self._counter[p]
            self.zero_grad()

        # print("group is",group)
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                        h = layer.weight.grad.data.transpose(1, 0)[idxs].flatten(1)
                    else:
                        w = layer.weight.data[idxs].flatten(1)
                        h = layer.weight.grad.data[idxs].flatten(1)

                    local_imp = (w**2 * h).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                
                if self.bias and layer.bias is not None and layer.bias.grad is not None:
                    b = layer.bias.data[idxs]
                    h = layer.bias.grad.data[idxs]
                    local_imp = (b**2 * h)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = (layer.weight).flatten(1)
                        h = (layer.weight.grad).flatten(1)
                    else:
                        w = (layer.weight).transpose(0, 1).flatten(1)
                        h = (layer.weight.grad).transpose(0, 1).flatten(1)

                    local_imp = (w**2 * h).sum(1)
                    if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                        local_imp = local_imp.repeat(layer.groups)
                    local_imp = local_imp[idxs]
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine:
                    if layer.weight.grad is not None:
                        w = layer.weight.data[idxs]
                        h = layer.weight.grad.data[idxs]
                        local_imp = (w**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None and layer.bias.grad is None:
                        b = layer.bias.data[idxs]
                        h = layer.bias.grad.data[idxs]
                        local_imp = (b**2 * h).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            
            # LN
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    if layer.weight.grad is not None:
                        w = layer.weight.data[idxs]
                        h = layer.weight.grad.data[idxs]
                        local_imp = (w**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None and layer.bias.grad is not None:
                        b = layer.bias.data[idxs]
                        h = layer.bias.grad.data[idxs]
                        local_imp = (b**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


# Aliases
class GroupNormImportance(MagnitudeImportance):
    pass

class GroupTaylorImportance(TaylorImportance):
    pass

class GroupHessianImportance(HessianImportance):
    pass