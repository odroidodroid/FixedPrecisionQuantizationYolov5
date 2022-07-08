import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn import Module, Parameter
import torch.nn.quantized.functional as F

from .quant_utils import *

def gcd(inputA, inputB) :
    while inputB:
        inputA, inputB = inputB, torch.fmod(inputA, inputB)
    return inputA


class QuantLinear(Module):
    """
    Class to quantize weights of given linear layer
    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(self,
                 weight_bit=8,
                 activation_bit=8,
                 bias_bit=None,
                 full_precision_flag=False,
                 quant_mode='symmetric',
                 per_channel=True,
                 fix_flag=False,
                 ):
        super().__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.activation_bit = activation_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode
        self.counter = 0

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        # self.register_buffer('weight_scale', torch.zeros(self.out_features))
        # self.register_buffer('weight_zeropoint', torch.zeros(self.out_features))
        self.weight = Parameter(linear.weight.data.clone())
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None
    
    def set_quant_param(self, save_path=None) :

        w = self.weight
        w_transform = w.data.contiguous().view(self.out_channels, -1)

        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        if self.quant_mode == 'symmetric' :
            self.weight_scale = symmetric_linear_quantization_params_weight(self.weight_bit, w_min, w_max, self.per_channel)

        
        elif self.quant_mode == 'asymmetric' :
            self.weight_scale, self.weight_zeropoint = asymmetric_linear_quantization_params(self.weight_bit, w_min, w_max, self.per_channel)
            

    def fix(self):
        self.fix_flag = True

    def unfix(self):
        self.fix_flag = False
    
    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        # quantize activation

        # perform the quantization
        if not self.full_precision_flag:
            if self.quant_mode == 'symmetric':
                x_transform = x.data.view(self.in_channels, -1)
                x_min, x_max = x_transform.min(dim=1).values, x_transform.max(dim=1).values

                x_scale = symmetric_linear_quantization_params_act(self.activation_bit, x_min, x_max)
                x_int = symmetric_linear_quantize(x, x_scale)

                weight_integer = self.weight_function(self.weight, self.weight_bit, self.weight_scale)
                y = ste_round.apply(F.linear(x_int, self.weight_integer))
                y_fp = symmetric_linear_dequantize(y, x_scale, x_zeropoint)
                return y_fp

            elif self.quant_mode == 'asymmetric':
                x_transform = x.data.view(self.in_channels, -1)
                x_min, x_max = x_transform.min(dim=1).values, x_transform.max(dim=1).values

                x_scale, x_zeropoint = asymmetric_linear_quantization_params(self.activation_bit, x_min, x_max)
                x_int = asymmetric_linear_quantize(x, x_scale, x_zeropoint)

                weight_integer = self.weight_function(self.weight_bit, self.weight_scale, self.weight_zeropoint)
                y = ste_round.apply(F.linear(x_int, weight=weight_integer))
                y_fp = asymmetric_linear_dequantize(y, x_scale, x_zeropoint)
                return y_fp


class QuantBatchNorm2d(Module) :
    def __init__(self, full_precision_flag=False,
                    quant_mode='asymmetric',
                    per_channel=True,
                    fix_flag=False,
                    fix_BN=False,
                    fix_BN_threshold=None):
        super().__init__()

        self.full_precision_flag=full_precision_flag
        self.per_channel=per_channel
        self.quant_mode = quant_mode
        self.fix_flag = fix_flag
        self.fix_BN = fix_BN
        self.training_BN_mode = fix_BN
        self.fix_BN_threshold = fix_BN_threshold

    def __repr__(self):
        s = super(QuantBatchNorm2d, self).__repr__()
        s = "(" + s + " )"
        return s



    def set_param(self, bn) :
        self.running_mean = bn.running_mean.data.clone()
        self.running_var = bn.running_var.data.clone()
        self.eps = bn.eps
        self.bn_weight = Parameter(bn.weight.data.clone())
        self.bn_bias = Parameter(bn.bias.data.clone())
        self.bn_momentum = 0.99

    def fix(self):
        """
        fix the BN statistics by setting fix_BN to True
        """
        self.fix_flag = True
        self.fix_BN = True

    def unfix(self):
        """
        change the mode (fixed or not) of BN statistics to its original status
        """
        self.fix_flag = False
        self.fix_BN = self.training_BN_mode

    def forward(self, x) :

        if not self.full_precision_flag :
            weight = self.bn_weight / (torch.sqrt(self.running_var + self.eps))
            bias = self.bn_bias - weight * self.running_mean

            x *= weight.view(1, -1, 1, 1)
            x += bias.view(1, -1, 1, 1)

            return x
        else :
            return x

class QuantMaxPool2d(Module):
    """
    Quantized MaxPooling Layer
    Parameters:
    ----------
    kernel_size : int, default 3
        Kernel size for max pooling.
    stride : int, default 2
        stride for max pooling.
    padding : int, default 0
        padding for max pooling.
    """

    def __init__(self,
                 kernel_size=3,
                 stride=2,
                 padding=0):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def __repr__(self):
        s = super(QuantMaxPool2d, self).__repr__()
        s = "(" + s + " )".format()
        return s

    def forward(self, x):
        x = self.pool(x)
        return x


class QuantDropout(Module):
    """
    Quantized Dropout Layer
    Parameters:
    ----------
    p : float, default 0
        p is the dropout ratio.
    """

    def __init__(self, p=0):
        super().__init__()

        self.dropout = nn.Dropout(p)

    def __repr__(self):
        s = super(QuantDropout, self).__repr__()
        s = "(" + s + " )"
        return s

    def forward(self, x):
        x = self.dropout(x)

        return x


class QuantUpsample(Module) :
    def __init__(self, size=None, scale_factor=2, mode='nearest') :
        super().__init__()
        
        self.upsample = nn.Upsample(size, scale_factor, mode)

    def __repr__(self):
        s = super(QuantUpsample, self).__repr__()
        s = "(" + s + " )"
        return s
    
    def forward(self, x) :
        x = self.upsample(x)
        return x

class QuantAveragePool2d(Module):
    """
    Quantized Average Pooling Layer
    Parameters:
    ----------
    kernel_size : int, default 7
        Kernel size for average pooling.
    stride : int, default 1
        stride for average pooling.
    padding : int, default 0
        padding for average pooling.
    """

    def __init__(self,
                 kernel_size=7,
                 stride=1,
                 padding=0):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.final_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def __repr__(self):
        s = super(QuantAveragePool2d, self).__repr__()
        s = "(" + s + " )"
        return s

    def set_param(self, pool):
        self.final_pool = pool

    def forward(self, x, x_scaling_factor=None):
        if type(x) is tuple:
            x_scaling_factor = x[1]
            x = x[0]

        if x_scaling_factor is None:
            return self.final_pool(x)

        x_scaling_factor = x_scaling_factor.view(-1)
        correct_scaling_factor = x_scaling_factor

        x_int = x / correct_scaling_factor
        x_int = ste_round.apply(x_int)
        x_int = self.final_pool(x_int)

        x_int = transfer_float_averaging_to_int_averaging.apply(x_int)

        return (x_int * correct_scaling_factor, correct_scaling_factor)

# class QuantBottleneck(Module) :
#     def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5) :
#         super().__init__()
#         self.c_ = int(out_channels * expansion)
#         self.cv1 = QuantConv2d(in_channels=in_channels, out_channels=self.c_, kernel=1, stride=1)
#         self.cv2 = QuantConv2d(in_channels=self.c_, out_channels=out_channels, kernel=3, stride=1, groups=groups)
#         self.add = shortcut and in_channels == out_channels

#     def forward(self, x) :
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class QuantConv2d(Module):
    """
    Class to quantize weights ofr given convolutional layer
    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(self, 
                weight_bit=8,
                activation_bit=8,
                bias_bit=8,
                full_precision_flag=False,
                quant_mode="symmetric",
                per_channel=True,
                fix_flag=False):
        super().__init__()

        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.activation_bit = activation_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)


    def __repr__(self):
        s = "QuantConv2d"
        s =  s + "(" + str(self.in_channels) + ", " + str(self.out_channels) + ", kernel_size=" + str(self.kernel_size) + ", stride=" + str(self.stride) + ")"
        return s

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        self.bias = Parameter(conv.bias.data.clone())

    def set_quant_param(self, save_path=None) :

        w = self.weight 
        b = self.bias
        if self.per_channel :
            w_transform = w.data.contiguous().view(self.out_channels, -1)
            w_min = w_transform.min(dim=1).values
            w_max = w_transform.max(dim=1).values
            b_transform = b.data.contiguous().view(self.out_channels, -1)
            b_min = b_transform.min(dim=1).values
            b_max = b_transform.max(dim=1).values

        else :
            w_min = w.data.min()
            w_max = w.data.max() 
            b_min = b.data.min()
            b_max = b.data.max()

        if self.quant_mode == 'symmetric' :
            self.weight_scale = symmetric_linear_quantization_params_weight(self.weight_bit, w_min, w_max)
            self.bias_scale = symmetric_linear_quantization_params_weight(self.weight_bit, b_min, b_max)

        elif self.quant_mode == 'asymmetric' :
            self.weight_scale, self.weight_zeropoint = asymmetric_linear_quantization_params(self.weight_bit, w_min, w_max)
            self.bias_scale, self.bias_zeropoint = asymmetric_linear_quantization_params(self.bias_bit, b_min, b_max)

    def fix(self):
        self.fix_flag = True

    def unfix(self):
        self.fix_flag = False

    def forward(self, x):

        # quantize activation

        if self.quant_mode == "symmetric":
            x_transform = x.data.contiguous().view(self.in_channels, -1)
            x_min, x_max = x_transform.min(), x_transform.max()
            
            x_scale = symmetric_linear_quantization_params_act(self.activation_bit, x_min, x_max)
            #x_int = symmetric_linear_quantize(x, x_scale)

            x_int = torch.quantize_per_tensor(x, scale=x_scale, zero_point=0, dtype=torch.quint8).to('cuda')


            #self.weight_function = SymmetricQuantFunction.apply
            #weight_integer = self.weight_function(self.weight, self.weight_bit, self.weight_scale)
            #bias_integer = self.weight_function(self.bias, self.weight_bit, self.bias_scale)

            weight_integer = torch.quantize_per_channel(self.weight, self.weight_scale, 
            torch.zeros(size=self.weight.shape).to('cuda'), 
            4, torch.quint8).to('cuda')
            bias_integer = torch.quantize_per_channel(self.bias, self.bias_scale, 
            torch.zeros(size=self.bias.shape).to('cuda').view(1, self.bias.shape[0], self.bias.shape[1], self.bias.shape[2]), 
            4, torch.quint8).to('cuda')

            y =  F.conv2d(x_int, weight_integer, bias_integer,
            self.stride, self.padding, self.dilation, self.groups, scale=self.weight_scale, zero_point=0) ### ste_round ??????
            #y_fp = symmetric_linear_dequantize_with_bias(y, x_scale, self.weight_scale, self.bias_scale, bias_integer)
            return y

        elif self.quant_mode =='asymmetric':
            x_transform = x.data.contiguous().view(self.in_channels, -1)
            x_min, x_max = x_transform.min(), x_transform.max()
            
            x_scale, x_zeropoint = asymmetric_linear_quantization_params(self.activation_bit, x_min, x_max)
            x_int = asymmetric_linear_quantize(x, x_scale, x_zeropoint)

            self.weight_function = AsymmetricQuantFunction.apply
            weight_integer = self.weight_function(self.weight, self.weight_bit, self.weight_scale, self.weight_zeropoint)
            bias_integer = self.weight_function(self.bias, self.bias_bit, self.bias_scale, self.bias_zeropoint)
            
            y =  ste_round.apply(F.conv2d(x_int, weight_integer, bias_integer, 
            self.stride, self.padding, self.dilation, self.groups))
            y_fp = asymmetric_linear_dequantize_with_bias(y, bias_integer, x_scale, self.weight_scale, self.bias_scale)
            return y_fp

        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))


def freeze_model(model):
    """
    freeze the activation range ### inference ####
    """
    if type(model) == QuantConv2d:
        model.fix()
    elif type(model) == QuantLinear:
        model.fix()
    elif type(model) == QuantBatchNorm2d:
        model.fix()
    elif type(model) == QuantUpsample :
        model.fix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            freeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                freeze_model(mod)


def unfreeze_model(model):
    """
    unfreeze the activation range ### training ###
    """
    if type(model) == QuantConv2d:
        model.unfix()
    elif type(model) == QuantBatchNorm2d :
        model.unfix()
    elif type(model) == QuantUpsample :
        model.unfix()
    elif type(model) == QuantLinear:
        model.unfix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            unfreeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                unfreeze_model(mod)



