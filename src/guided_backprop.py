"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
@editor: Caner Ozer - github.com/canerozer
"""
import argparse
import yaml
import os
import numpy as np
from PIL import Image

import torch
from torch.nn import ReLU, Sequential
from torchvision.models.resnet import BasicBlock, Bottleneck


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, name):
        self.model = model
        self.name = name

        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Gather the first layer depending on the architecture
        if "alexnet" in self.name:
            first_layer = list(self.model.features._modules.items())[0][1]
        elif "resnet" in self.name:
            first_layer = self.model.conv1
        else:
            raise NotImplementedError

        # Register hook to the first layer
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        # Loop through layers, hook up ReLUs
        if "alexnet" in self.name:
            self.hook_alexnet_relu()
        elif "resnet" in self.name:
            self.hook_resnet_relu()
        else:
            raise NotImplementedError

    def relu_bw_hook_f(self, module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero
        """
        # Get last forward output
        corresponding_forward_output = self.forward_relu_outputs[-1]
        corresponding_forward_output[corresponding_forward_output > 0] = 1
        modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
        del self.forward_relu_outputs[-1]  # Remove last forward output
        return (modified_grad_out,)

    def relu_fw_hook_f(self, module, ten_in, ten_out):
        """
        Store results of forward pass
        """
        self.forward_relu_outputs.append(ten_out)

    def hook_alexnet_relu(self):
        relu_bw_hook_f = self.relu_bw_hook_f
        relu_fw_hook_f = self.relu_fw_hook_f

        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_bw_hook_f)
                module.register_forward_hook(relu_fw_hook_f)

    def hook_resnet_relu(self):
        relu_bw_hook_f = self.relu_bw_hook_f
        relu_fw_hook_f = self.relu_fw_hook_f

        for pos, module in self.model._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_bw_hook_f)
                module.register_forward_hook(relu_fw_hook_f)
            elif isinstance(module, Sequential):
                for bpos, block in module._modules.items():
                    if isinstance(block, (Bottleneck, BasicBlock)):
                        for lpos, layer in block._modules.items():
                            if isinstance(module, ReLU):
                                module.register_backward_hook(relu_bw_hook_f)
                                module.register_forward_hook(relu_fw_hook_f)

    def generate_gradients(self, input_image, target_class):
        device = input_image.device
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        one_hot_output = one_hot_output.to(device)
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr



