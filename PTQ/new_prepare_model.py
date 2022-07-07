from abc import abstractclassmethod
from utils.quant_utils.new_quant_module import *
from utils.quant_utils.new_quant_utils import *
from models.common import *
from models.yolo import Model, Detect
from copy import deepcopy
### where to hook
### conv, fused_conv, fc


### weight matching ???? keys <- weight

# def prepare_model_with_hook(model=None, bit_width=None, mode=None, save_path=None) :

#     hooks, hook_handles = [], []
#     for n, m in (model.named_modules()) : 
#         if type(m) == nn.Conv2d : 
#             hook = input_conv_hook(m, n, bit_width, mode, save_path)
#             hooks.append(hook) 
#             hook_handles.append(m.register_forward_hook(hook.hook))
#         elif type(m) == nn.Linear :
#             hook = input_linear_hook(m, n, bit_width, mode, save_path)
#             hooks.append(hook)
#             hook_handles.append(m.register_forward_hook(hook.hook))
#         elif type(m) == nn.BatchNorm2d :
#             hook == input_batchnorm_hook(m, n, save_path)
#             hooks.append(hook)
#             hook_handles.append(m.register_forward_hook(hook.hook))
#         elif type(m) == nn.MaxPool2d :
#             hook = input_maxpool_hook(m, n, save_path)
#             hooks.append(hook)
#             hook_handles.append(m.register_forward_hook(hook.hook))
#         elif type(m) == nn.Upsample :
#             hook = input_upsample_hook(m, n, save_path)
#             hooks.append(hook)
#             hook_handles.append(m.register_forward_hook(hook.hook))
#         else :
#             hook = m
#             hooks.append(hook)
#             hook_handles.append(m.register_forward_hook(hook.hook))        

#     return model
 

def prepare_model (model=None, bit_width=None, mode=None, save_path=None, calibrator=None, axis=None, unsigned=None) :

    # define model parameter
    # prepare_model
    #    -> bit-width selection preparation
    #    -> build quantization model
    #    -> weight quantization ? 

    for attr_outer in dir(model):
        if attr_outer == 'model':
            mod_outer = getattr(model, attr_outer)
            for attr in dir(mod_outer) :
                if attr == 'model' :
                    mod = getattr(mod_outer, attr)
                    prepare_module(mod, attr_outer + '.' + attr, bit_width, mode, save_path, calibrator, axis, unsigned)

    return model
    
def prepare_module(mod=None, attr=None, bit_width=None, mode=None, save_path=None, calibrator=None, axis=None, unsigned=None) :

    if isinstance(mod, nn.Sequential)  :
        for n, m in mod.named_children() :
            prepare_module(m, attr + '.' + n, bit_width, mode, save_path, calibrator, axis, unsigned)
               
    elif isinstance(mod, Conv) :
        mod_copy = deepcopy(mod)
        for n, m in mod_copy.named_children() :
            quant_mod = quantize_module(m, attr + '.' + n, bit_width, mode, save_path, calibrator, axis, unsigned)
            setattr(mod, n, quant_mod)

    elif isinstance(mod, Bottleneck) :
        for n, m in mod.named_children() :
            prepare_module(m, attr + '.' + n, bit_width, mode, save_path, calibrator, axis, unsigned)

    elif isinstance(mod, C3) :
        for n, m in mod.named_children() :
            prepare_module(m, attr + '.' + n, bit_width, mode, save_path, calibrator, axis, unsigned)
            
    elif isinstance(mod, SPPF) :
        for n, m in mod.named_children() :
            prepare_module(m, attr + '.' + n, bit_width, mode, save_path, calibrator, axis, unsigned)
    
    elif isinstance(mod,Detect) :
        for n, m in mod.named_children() :
            prepare_module(m, attr + '.' + n, bit_width, mode, save_path, calibrator, axis, unsigned)
    
    elif isinstance(mod, nn.ModuleList) :
        for n, m in mod.named_children() :
            prepare_module(m, attr + '.' + n, bit_width, mode, save_path, calibrator, axis, unsigned)

def quantize_module(mod=None, attr=None, bit_width=None, mode=None, save_path=None, calibrator=None, axis=None, unsigned=None) :

    if isinstance(mod, nn.Conv2d) :
        quant_mod = QuantConv2d(weight_bit=bit_width, activation_bit=bit_width, quant_mode=mode,
        calibrator=calibrator, axis=axis, unsigned=unsigned)
        quant_mod.set_param(mod)
        quant_mod.set_quant_param(save_path=save_path)
        return quant_mod
        
    elif isinstance(mod, nn.Linear) :
        quant_mod = QuantLinear(weight_bit=bit_width, activation_bit=bit_width, quant_mode=mode)
        quant_mod.set_param(mod)
        quant_mod.set_quant_param(save_path=save_path)
        return quant_mod

    else :
        return mod


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
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            freeze_model(m)
    # else:
    #     for attr in dir(model):
    #         mod = getattr(model, attr)
    #         if isinstance(mod, nn.Module) and 'norm' not in attr:
    #             freeze_model(mod)


def unfreeze_model(model):
    """
    unfreeze the activation range ### training ###
    """
    if type(model) == QuantConv2d:
        model.unfix()
    elif type(model) == QuantLinear:
        model.unfix()
    elif type(model) == QuantBatchNorm2d:
        model.unfix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            unfreeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                unfreeze_model(mod)