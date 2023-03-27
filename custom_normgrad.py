"""
Created on Tue Sep 8 11:06:51 2020

@author: Sylvestre Rebuffi - github.com/srebuffi
@editor: Caner Ozer - github.com/canerozer
"""
import argparse
import yaml
import os
import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import Sequential
from torch.nn import functional as F

from src.misc_functions import (get_example_params, DictAsMember,
                               open_image, preprocess_image,
                               recreate_image, show_bbox,
                               custom_save_class_activation_images,
                               custom_save_np_arr,
                               get_boxes, get_mask, fit_bbox,
                               blur_input_tensor,
                               get_module_by_name)
from src.models import _get_classification_model as get_model


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
                if isinstance(module, nn.ReLU):
                    recursive_setattr(model, name, nn.ReLU(inplace=False))
                elif isinstance(module, nn.SiLU):
                    recursive_setattr(model, name, nn.SiLU(inplace=False))

        anti_inplace(self.model)
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.handlers.append(module.register_forward_hook(fw_hook(name)))
                self.handlers.append(module.register_full_backward_hook(bw_hook(name)))

                
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
        
    def __call__(self, input_image, phase1, target_class=None, device=None):
        input_image = input_image.to(device)
        output = self.extractor(input_image)
        if "cuda" in str(output.device):
            output = output.cpu()
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
        if "efficientnet" in self.model_name.lower():
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()

        # Backward pass with specified target
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Process all the targetted layers
        for num, target_layer in enumerate(self.target_layers):
            target = self.extractor.fmap_pool[target_layer]
            grad_init = self.extractor.grad_pool[target_layer]

            target = target.to(device)
            grad_init = grad_init.to(device)
            
            # Phase 1
            if phase1 == "conv1x1":
                grad_init = torch.norm(grad_init, 2, 1, keepdim=True)
                target = torch.norm(target, 2, 1, keepdim=True)
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
                unfold_act = torch.norm(unfold_act, 2, 1, keepdim=True)
                grad_init = torch.norm(grad_init, 2, 1, keepdim=True)
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
            out = (out - torch.min(out)) / (torch.max(out) - torch.min(out) + 1e-8)
            
            #TODO: Reconsider using continue here.
            if torch.all(out == 0):
                num -= 1
                final_out = torch.ones_like(out).cpu().data.numpy()[0, 0]
                continue

            if num == 0:
                final_out = out.cpu().data.numpy()[0, 0]
            else:
                final_out *= out.cpu().data.numpy()[0, 0]
            
        final_out = (final_out - np.min(final_out)) / (np.max(final_out) - np.min(final_out) + 1e-8)

        return final_out, pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize NormGrad')
    parser.add_argument('--yaml', type=str, metavar="YAML",
                        default="configs/R50_LVOT.yaml",
                        help='Enter the path for the YAML config')
    parser.add_argument('--img', type=str, metavar='I', default=None,
                        help="Enter the image path")
    parser.add_argument('--target', type=int, metavar='T', default=None,
                        help="Enter the target class ID")
    parser.add_argument('--pg-mode', type=str, metavar='PGM', default="singlepeak",
                        help='Enter the mode for Pointing Game')
    parser.add_argument('--device', type=str, metavar="D", default="cpu",
                        help='cpu or cuda')
    parser.add_argument('--model-path', type=str, metavar="MD", default=None,
                        help="Enter the path for the model file")
    parser.add_argument('--tau', type=int, metavar='TAU', default=None,
                        help='Enter the TAU value for Pointing Game')
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--randomized', action='store_true')
    parser.add_argument('--reprod-id', type=int, metavar='RI', default=0,
                        help="Enter the reproduction ID number, if evaluated for a different model instance")
    parser.add_argument('--cuda', action='store_true')             
    parser.add_argument('--debug', action='store_true')      
    args = parser.parse_args()

    device = args.device
    print(device)

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

    if args.debug:
         torch.set_printoptions(profile="full")

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

    alpha = vis_args.RESULTS.alpha
    h = w = vis_args.DATASET.size

    ll = vis_args.MODEL.last_layer
    lb = vis_args.MODEL.last_block
    if isinstance(ll, list) and isinstance(lb, list):
        if len(ll) != len(lb):
            raise IndexError("lengths of vis_args.MODEL.last_layer and \
                              vis_args.MODEL.last_block do not match!")
    # Initialize NormGrad
    normgrad = NormGrad(pretrained_model, vis_args.MODEL,
                        input_size=(h, w))
    try:
        phase1 = vis_args.MODEL.phase1
    except KeyError:        
        phase1 = "scaling"
    iscombined = "combined" if len(ll) > 0 and isinstance(ll, list) else "single"
    smooth_cond = "smoothed" if vis_args.RESULTS.POINTING_GAME.SMOOTHING.state else "unsmoothed"
    if args.randomized:
        if args.no_pretrained:
            random_cond = "fullyrandomized"
        else:
            random_cond = "randomized"
    else:
        random_cond = "pretrained"
    reprod_id = "reprod" + str(args.reprod_id) if args.reprod_id > 0 else "" 
    obj = "_".join(["normgrad", vis_args.MODEL.name, iscombined, phase1, random_cond, smooth_cond])
    if reprod_id != "":
        obj += "_" + reprod_id

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

    preds_dict = {}
    pg_dict = {}

    d = 0
    for f, path in tqdm.tqdm(zip(files, paths)):
        img = open_image(path)
        prep_img = preprocess_image(img, h=h, w=w)
        cam, pred = normgrad(prep_img, phase1, vis_args.DATASET.target_class, device=device)
        cam_orig = cam.copy()
  
        # Run Pointing Game
        if vis_args.RESULTS.POINTING_GAME.state:
            cam = cam_orig
            if vis_args.RESULTS.POINTING_GAME.SMOOTHING.state:
                sigma = vis_args.RESULTS.POINTING_GAME.SMOOTHING.sigma
                cam = torch.Tensor(cam[None])
                cam = blur_input_tensor(cam, sigma=sigma).numpy().squeeze()

            if vis_args.DATASET.MASK.state:
                mask = get_mask(f, vis_args.DATASET.MASK.path, size=(h,w))
                boxes = fit_bbox(mask)
            elif vis_args.DATASET.BBOX.state:
                mask = None
                boxes = get_boxes(vis_args.DATASET.BBOX.path, f, img,
                                  size=(h,w))

            hit = pg.evaluate(cam, mask=mask, boxes=boxes)
            _ = pg.accumulate(hit[0], 1)
            pg_dict[f] = hit[0]

        if vis_args.RESULTS.DRAW_GT_BBOX.state:
            cam = show_bbox(img, cam, f,
                            bbox_gt_src=vis_args.DATASET.BBOX.path,
                            mask_gt_src=vis_args.DATASET.MASK.path)
        img = img.resize((h, w))

        custom_save_class_activation_images(img, cam,
                                            vis_args.RESULTS.dir, f,
                                            obj=obj, alpha=alpha
                                           )
        preds_dict[f] = pred

        d += 1

    # Show results for Pointing Game
    if vis_args.RESULTS.POINTING_GAME.state:
        print(pg.print_stats())

    img_names, preds = list(preds_dict.keys()), list(preds_dict.values())
    df = pd.DataFrame({"image_names": img_names, "predictions": preds})
    df.to_csv(vis_args.RESULTS.save_preds_to)

    pg_img_names, pg_preds = list(pg_dict.keys()), list(pg_dict.values())
    pg_df = pd.DataFrame({"image_names": pg_img_names, "predictions": pg_preds})
    pg_df_path = os.path.abspath(os.path.join(vis_args.RESULTS.save_preds_to, '..'))
    pg_df.to_csv(os.path.join(pg_df_path, obj, 'pg.csv'))

    print('NormGrad completed')
