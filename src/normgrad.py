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
import torch.nn.functional as F
from torch.nn import Sequential
from torchvision.models.resnet import Bottleneck, BasicBlock

class FeatureExtract:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.grads = []
        self.fmap_pool = {}
        self.grad_pool = {}
        self.handlers = []
        
        def fw_hook(key):
            def _forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()
            return _forward_hook
            
        def bw_hook(key):
            def _backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()
            return _backward_hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.handlers.append(module.register_forward_hook(fw_hook(name)))
                self.handlers.append(module.register_backward_hook(bw_hook(name)))
                
    def __call__(self, x):
        return self.model(x)
        
        
class NormGrad:
    """
    Produces class activation map
    """
    def __init__(self, model, model_args, input_size=None):
        self.model = model
        self.model.eval()
        self.model_name = model_args.name
        target_layer = model_args.last_layer
        target_block = model_args.last_block
        if isinstance(target_layer, list) and isinstance(target_block, list):
            self.target_layers = [l + "." + str(b) for l, b in zip(target_layer, target_block)]
        else:
            self.target_layers = [target_layer + "." + str(target_block)]
        self.extractor = FeatureExtract(self.model, self.target_layers)
        self.size = input_size
            
    def forward(self, input):
        return self.model(input)
        
    def __call__(self, input_image, phase1, target_class=None):
        output = self.extractor(input_image)
        pred = np.argmax(output.data.numpy())
        if target_class is None:
            target_class = pred

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # Zero grads
        if self.model_name == "alexnet":
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
        if "resnet" in self.model_name:
            for name, module in self.model._modules.items():
                module.zero_grad()
                
        # Backward pass with specified target
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Process all the targetted layers
        for num, target_layer in enumerate(self.target_layers):
            target = self.extractor.fmap_pool[target_layer].cpu()
            grad_init = self.extractor.grad_pool[target_layer].cpu()
            
            # Phase 1
            if phase1 == "conv1x1":
                out = -torch.matmul(target.permute(0, 2, 3, 1).view(
                                    -1, target.size(1), 1),
                                    grad_init.permute(0, 2, 3, 1).view(
                                    -1, 1, grad_init.size(1)))
                out = out.view(out.size(0), -1).permute(1, 0).view(
                        1, -1, target.shape[2], target.shape[3])
            elif phase1 == "conv3x3":
                unfold_act = F.unfold(target, kernel_size=3, padding=1)
                unfold_act = unfold_act.view(1, target.shape[1] * 9,
                                             target.shape[2], target.shape[3])
                out = -torch.matmul(unfold_act.permute(0, 2, 3, 1).view(
                                    -1, unfold_act.size(1), 1),
                                    grad_init.permute(0, 2, 3, 1).view(
                                    -1, 1, grad_init.size(1)))
                out = out.view(out.size(0), -1).permute(1, 0).view(
                        1, -1, target.shape[2], target.shape[3])
            elif phase1 == "conv3x3_depthwise":
                unfold_act = F.unfold(target, kernel_size=3, padding=1)
                unfold_act = unfold_act.view(1, target.shape[1] * 9,
                             target.shape[2], target.shape[3])
                out = -(unfold_act * grad_init[:, :, None]).view(
                        -1, 1, target.shape[2], target.shape[3])
            elif phase1 == "scaling":
                out = -target * grad_init
            elif phase1 == "bias":
                if target_layer == "layer4":
                    continue    # Ommiting the latest bias layer of ResNet
                out = -grad_init

            # Phase 2 (Normalization)
            out = torch.norm(out, 2, 1, keepdim=True)

            out = F.interpolate(out, size=self.size,
                                mode="bilinear")
            out = (out - torch.min(out)) / (torch.max(out) - torch.min(out))
            
            if num == 0:
                final_out = out.cpu().data.numpy()[0, 0]
            else:
                final_out *= out.cpu().data.numpy()[0, 0]
                
        return final_out, pred

