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

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, model_name, target_layer=None, target_block=None):
        self.model = model
        self.model_name = model_name
        self.target_layer = target_layer
        self.target_block = str(target_block)

        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def hook_alexnet(self, conv_output, x):
        for mpos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(mpos) == self.target_layer:
                x.register_hook(self.save_gradient)
                # Save the convolution output of the target layer
                conv_output = x  
        return conv_output, x

    def hook_resnet(self, conv_output, x):
        for mname, mmodule in self.model._modules.items():
            if mname == "fc":
                break
            if mname == self.target_layer and \
            isinstance(mmodule, Sequential):
                for id, block in mmodule._modules.items():
                    if id == self.target_block:
                        for name, module in block._modules.items():
                            x = module(x)
                            if name == "conv3" and \
                            isinstance(block, Bottleneck):
                                x.register_hook(self.save_gradient)
                                conv_output = x
                            elif name == "conv2" and \
                            isinstance(block, BasicBlock):
                                x.register_hook(self.save_gradient)
                                conv_output = x
                    else:
                        x = block(x)
            else:
                x = mmodule(x)
        return conv_output, x

    def hook_sononet(self, conv_output, x):
        conv_output = [None, None, None]
        for mname, mmodule in self.model._modules.items():
            if mname == "conv3":
                for name, module in mmodule._modules.items():
                    x = module(x)
                    if name == "conv3":
                        conv_output[0] = x
            elif mname == "conv4":
                for name, module in mmodule._modules.items():
                    x = module(x)
                    if name == "conv2":
                        conv_output[1] = x
            elif mname == "conv5":
                for name, module in mmodule._modules.items():
                    x = module(x)
                    if name == "conv2":
                        conv_output[2] = x
                        x.register_hook(self.save_gradient)
            elif mname == "classifier" or "compatibility_score" in mname:
                break   
            else:
                x = mmodule(x)      
        return conv_output, x

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        if self.model_name == "alexnet":
            conv_output, x = self.hook_alexnet(conv_output, x)
        if "resnet" in self.model_name:
            conv_output, x = self.hook_resnet(conv_output, x)
        if "sononet" in self.model_name:
            conv_output, x = self.hook_sononet(conv_output, x)
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # Forward pass on the classifier
        if self.model_name == "alexnet":
            x = x.view(x.size(0), -1)  # Flatten
            x = self.model.classifier(x)
        if "resnet" in self.model_name:
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
        if "sononet" in self.model_name:
            conv3, conv4, conv5 = conv_output
            batch_size = x.shape[0]
            pooled = F.adaptive_avg_pool2d(conv5, (1, 1)).view(batch_size, -1)

            # Attention Mechanism
            g_conv1, att1 = self.model.compatibility_score1(conv3, conv5)
            g_conv2, att2 = self.model.compatibility_score2(conv4, conv5)

            # flatten to get single feature vector
            fsizes = self.model.attention_filter_sizes
            g1 = torch.sum(g_conv1.view(batch_size, fsizes[0], -1), dim=-1)
            g2 = torch.sum(g_conv2.view(batch_size, fsizes[1], -1), dim=-1)

            x = self.model.aggregate(g1, g2, pooled)
            conv_output = conv_output[2]

        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    #def __init__(self, model, target_layer):
    def __init__(self, model, model_args):
        self.model = model
        self.model.eval()
        self.model_name = model_args.name
        target_layer = model_args.last_layer
        target_block = model_args.last_block
        # Define extractor
        self.extractor = CamExtractor(self.model, self.model_name,
                                      target_layer=target_layer,
                                      target_block=target_block)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        pred = np.argmax(model_output.data.numpy())
        if target_class is None:
            target_class = pred

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # Zero grads
        if self.model_name == "alexnet":
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
        if "resnet" in self.model_name:
            for name, module in self.model._modules.items():
                module.zero_grad()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]

        # Get convolution outputs
        target = conv_output.data.numpy()[0]

        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient

        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)

        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam, pred

