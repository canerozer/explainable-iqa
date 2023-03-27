"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
@editor: Caner Ozer - github.com/canerozer
"""
import argparse
import yaml
import os
import tqdm
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.nn import ReLU, SiLU, Sequential
from torchvision.models.resnet import BasicBlock, Bottleneck
from src.misc_functions import (get_example_params,
                                convert_to_grayscale,
                                get_positive_negative_saliency,
                                DictAsMember, open_image,
                                custom_save_np_arr,
                                preprocess_image, custom_save_gradient_images,
                                custom_save_class_activation_images,
                                get_boxes, get_mask, fit_bbox, blur_input_tensor)
from src.models import _get_classification_model as get_model


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, name, device=None):
        self.model = model
        self.name = name
        self.device = device
        if self.device != torch.device('cpu'):
            self.iscuda = True
        else:
            self.iscuda = False

        self.gradients = None
        self.forward_relu_outputs = []
        self.forward_silu_outputs = []

        # Put model in evaluation mode
        self.model.eval()
        if "efficientnet" in self.name.lower():
            self.update_silus()
        else:
            self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Big thanks to mawilson @ hugginhface forums
        # https://discuss.huggingface.co/t/how-can-i-replace-modules-in-a-pretrained-model/16338
        def recursive_setattr(obj, attr, value):
            attr = attr.split('.', 1)
            if len(attr) == 1:
                setattr(obj, attr[0], value)
            else:
                recursive_setattr(getattr(obj, attr[0]), attr[1], value)

        def anti_inplace(model):
            for name, module in model.named_modules():
                if isinstance(module, ReLU):
                    recursive_setattr(model, name, ReLU(inplace=False))
                elif isinstance(module, SiLU):
                    recursive_setattr(model, name, SiLU(inplace=False))

        anti_inplace(self.model)

        # Gather the first layer depending on the architecture
        if "alexnet" in self.name:
            first_layer = list(self.model.features._modules.items())[0][1]
        elif "resnet" in self.name:
            first_layer = self.model.conv1
        elif "efficientnet" in self.name.lower():
            first_layer = self.model.features[0][0]
        else:
            raise NotImplementedError

        # Register hook to the first layer
        first_layer.register_full_backward_hook(hook_function)

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

    def update_silus(self):
        self.hook_efficientnet_silu()

    def silu_bw_hook_f(self, module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero
        """
        # Get last forward output
        def silu_bw(x):
            return torch.sigmoid(x) + x * torch.sigmoid(x) * (1 - torch.sigmoid(x))

        corresponding_forward_output = self.forward_silu_outputs[-1]
        corresponding_forward_output = silu_bw(corresponding_forward_output)
        deconv_out = silu_bw(grad_in[0])
        modified_grad_out = corresponding_forward_output * deconv_out * grad_in[0]
        
        del self.forward_silu_outputs[-1]  # Remove last forward output
        return (modified_grad_out,)

    def silu_fw_hook_f(self, module, ten_in, ten_out):
        """
        Store results of forward pass
        """
        self.forward_silu_outputs.append(ten_out)

    def hook_efficientnet_silu(self):
        silu_bw_hook_f = self.silu_bw_hook_f
        silu_fw_hook_f = self.silu_fw_hook_f

        for module in self.model.modules():
            if isinstance(module, SiLU):
                module.register_backward_hook(silu_bw_hook_f)
                module.register_forward_hook(silu_fw_hook_f)

    def generate_gradients(self, input_image, target_class):
        device = input_image.device
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        if self.iscuda:
            one_hot_output = one_hot_output.cuda()
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        if self.iscuda:
            gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        else:
            gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Guided Backprop')
    parser.add_argument('--yaml', type=str, metavar="YAML",
                        default="configs/R50_LVOT.yaml",
                        help='Enter the path for the YAML config')
    parser.add_argument('--img', type=str, metavar='I', default=None,
                        help="Enter the image path")
    parser.add_argument('--target', type=int, metavar='T', default=None,
                        help="Enter the target class ID")
    parser.add_argument('--pg-mode', type=str, metavar='PGM', default="singlepeak",
                        help='Enter the mode for Pointing Game')
    parser.add_argument('--model-path', type=str, metavar="MD", default=None,
                        help="Enter the path for the model file")
    parser.add_argument('--tau', type=int, metavar='TAU', default=None,
                        help='Enter the TAU value for Pointing Game')
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--randomized', action='store_true')
    parser.add_argument('--reprod-id', type=int, metavar='RI', default=0,
                        help="Enter the reproduction ID number, if evaluated for a different model instance")
    parser.add_argument('--cuda', action='store_true')  
    args = parser.parse_args()

    yaml_path = args.yaml
    with open(yaml_path, 'r') as f:
        vis_args = DictAsMember(yaml.safe_load(f))

    if args.img:
        vis_args["DATASET"]["path"] = args.img
        vis_args["DATASET"]["target_class"] = args.target

    if args.model_path:
        vis_args["MODEL"]["path"] = args.model_path
        print(f"Using model path at {vis_args.MODEL.path}")

    if args.no_pretrained:
        vis_args["MODEL"]["pretrained"] = False
        print("Not using any pretrained model")
    else:
        print("Using the ImageNet pretrained model")

    if args.tau:
        vis_args["RESULTS"]["POINTING_GAME"]["tolerance"] = args.tau

    # Load model & pretrained params
    pretrained_model = get_model(vis_args.MODEL)
    if not args.randomized:
        print("Using pretrained model")
        state = torch.load(vis_args.MODEL.path)
        try:
            pretrained_model.load_state_dict(state["model"])
        except KeyError as e:
            pretrained_model.load_state_dict(state)
    else:
        print("Using model with random parameters")

    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    if args.cuda:
        pretrained_model = pretrained_model.cuda()

    alpha = vis_args.RESULTS.alpha
    h = w = vis_args.DATASET.size

    # Initialize GBP
    GBP = GuidedBackprop(pretrained_model, vis_args.MODEL.name, device=device)

    # Get filenames and create absolute paths
    if os.path.isdir(vis_args.DATASET.path):
        files = os.listdir(vis_args.DATASET.path)
        paths = [os.path.join(vis_args.DATASET.path, f) for f in files]
    elif os.path.isfile(vis_args.DATASET.path):
        files = list(vis_args.DATASET.path.split("/")[-1])
        paths = [vis_args.DATASET.path]

    # Poining Game Constructor
    if vis_args.RESULTS.POINTING_GAME.state:
        from src.pointing_game import PointingGame
        n_classes = vis_args.MODEL.n_classes
        tolerance = vis_args.RESULTS.POINTING_GAME.tolerance
        pg = PointingGame(n_classes, tolerance=tolerance)

    pg_dict = {}

    d = 0
    for f, path in tqdm.tqdm(zip(files, paths)):
        # if d == 10:
        #     break
        img = open_image(path)
        prep_img = preprocess_image(img, h=h, w=w)
        if args.cuda:
            prep_img = prep_img.to(device)
        guided_grads = GBP.generate_gradients(prep_img,
                                              vis_args.DATASET.target_class)

        pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
        bw_pos_sal = convert_to_grayscale(pos_sal)
        bw_neg_sal = convert_to_grayscale(neg_sal)
        bw_guided_grads = convert_to_grayscale(guided_grads)

        if vis_args.RESULTS.POINTING_GAME.state:
            if vis_args.RESULTS.POINTING_GAME.SMOOTHING.state:
                sigma = vis_args.RESULTS.POINTING_GAME.SMOOTHING.sigma
                bw_pos_sal = torch.Tensor(bw_pos_sal[None])
                bw_pos_sal = blur_input_tensor(bw_pos_sal, sigma=sigma).numpy()[0]
            if vis_args.DATASET.MASK.state:
                mask = get_mask(f, vis_args.DATASET.MASK.path, size=(h,w))
                boxes = fit_bbox(mask)
            elif vis_args.DATASET.BBOX.state:
                mask = None
                boxes = get_boxes(vis_args.DATASET.BBOX.path, f, img,
                                  size=(h,w))
            hit = pg.evaluate(bw_pos_sal.squeeze(), mask=mask, boxes=boxes)
            _ = pg.accumulate(hit[0], 1) #####
            pg_dict[f] = hit[0]

        smooth_cond = "smoothed" if vis_args.RESULTS.POINTING_GAME.SMOOTHING.state else "unsmoothed"
        if args.randomized:
            if args.no_pretrained:
                random_cond = "fullyrandomized"
            else:
                random_cond = "randomized"
        else:
            random_cond = "pretrained"
        reprod_id = "reprod" + str(args.reprod_id) if args.reprod_id > 0 else "" 
        obj1 = "_".join(["bw_gbp_pos", vis_args.MODEL.name, smooth_cond, random_cond])
        obj2 = "_".join(["bw_gbp_neg", vis_args.MODEL.name, smooth_cond, random_cond])
        obj3 = "_".join(["bw_gbp", vis_args.MODEL.name, smooth_cond, random_cond])
        if reprod_id != "":
            obj1 += "_" + reprod_id
            obj2 += "_" + reprod_id
            obj3 += "_" + reprod_id

        prep_img = prep_img[0].mean(dim=0, keepdim=True)
        if args.cuda:
            prep_img = prep_img.cpu()

        img = img.resize((h, w))
        custom_save_class_activation_images(img, bw_pos_sal[0],
                                            vis_args.RESULTS.dir, f,
                                            obj=obj1, alpha=alpha
                                           )
        custom_save_class_activation_images(img, bw_neg_sal[0],
                                            vis_args.RESULTS.dir, f,
                                            obj=obj2, alpha=alpha
                                           )
        custom_save_class_activation_images(img, bw_guided_grads[0],
                                            vis_args.RESULTS.dir, f,
                                            obj=obj3, alpha=alpha
                                           )


        d += 1

    if vis_args.RESULTS.POINTING_GAME.state:
        print(pg.print_stats())

    pg_img_names, pg_preds = list(pg_dict.keys()), list(pg_dict.values())
    pg_df = pd.DataFrame({"image_names": pg_img_names, "predictions": pg_preds})
    pg_df_path = os.path.abspath(os.path.join(vis_args.RESULTS.save_preds_to, '..'))
    pg_df.to_csv(os.path.join(pg_df_path, obj1, 'pg.csv'))

    print('Guided backprop completed')
