"""
Created on Thu Oct 26 11:06:51 2017

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
import torch.nn as nn
from torch.nn import Sequential
from torchvision.models.resnet import Bottleneck, BasicBlock

from src.misc_functions import (get_example_params, DictAsMember,
                                open_image, preprocess_image,
                                recreate_image, show_bbox,
                                custom_save_class_activation_images,
                                custom_save_np_arr,
                                get_boxes, get_mask, fit_bbox,
                                blur_input_tensor)
from src.models import _get_classification_model as get_model
import torch.nn.functional as F


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

    def hook_efficientnet(self, conv_output, x):
        for mname, mmodule in self.model._modules.items():
            if mname == "classifier":
                break
            if mname == "features" and isinstance(mmodule, Sequential):
                for id, block in mmodule._modules.items():
                    if id == self.target_layer.split(".")[-1]:
                        for name, module in block._modules.items():
                            x = module(x)
                            if name == self.target_block:
                                x.register_hook(self.save_gradient)
                                conv_output = x
                    else:
                        x = block(x)

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
        if "efficientnet" in self.model_name.lower():
            conv_output, x = self.hook_efficientnet(conv_output, x)
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
        if "efficientnet" in self.model_name.lower():
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)

        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    #def __init__(self, model, target_layer):
    def __init__(self, model, model_args, device=None):
        self.model = model
        self.model.eval()
        self.model_name = model_args.name
        target_layer = model_args.last_layer
        target_block = model_args.last_block
        self.device = device
        if self.device != torch.device('cpu'):
            self.iscuda = True
        else:
            self.iscuda = False
        # Define extractor
        self.extractor = CamExtractor(self.model, self.model_name,
                                      target_layer=target_layer,
                                      target_block=target_block)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        # pred = np.argmax(model_output.data.numpy())
        pred = torch.argmax(model_output)
        if target_class is None:
            target_class = pred

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        if self.iscuda:
            one_hot_output = one_hot_output.cuda()

        # Zero grads
        if self.model_name == "alexnet":
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
        if "resnet" in self.model_name:
            for name, module in self.model._modules.items():
                module.zero_grad()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        # Get hooked gradients and conv outputs
        if self.iscuda:
            guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
            target = conv_output.data.cpu().numpy()[0]
        else:
            guided_gradients = self.extractor.gradients.data.numpy()[0]
            target = conv_output.data.numpy()[0]

        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient

        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)

        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.Resampling.LANCZOS))/255
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


if __name__ == '__main__':
    import sys
    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)

    parser = argparse.ArgumentParser(description='Visualize Guided GradCAM')
    parser.add_argument('--yaml', type=str, metavar="YAML",
                        default="configs/R50_LVOT.yaml",
                        help='Enter the path for the YAML config')
    parser.add_argument('--img', type=str, metavar='I', default=None,
                        help="Enter the image path")
    parser.add_argument('--target', type=int, metavar='T', default=None,
                        help="Enter the target class ID")
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

    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    if args.cuda:
        pretrained_model = pretrained_model.cuda()

    # Initialize GradCam and GBP
    gcv2 = GradCam(pretrained_model, vis_args.MODEL, device=device)

    smooth_cond = "smoothed" if vis_args.RESULTS.POINTING_GAME.SMOOTHING.state else "unsmoothed"
    if args.randomized:
        if args.no_pretrained:
            random_cond = "fullyrandomized"
        else:
            random_cond = "randomized"
    else:
        random_cond = "pretrained"
    reprod_id = "reprod" + str(args.reprod_id) if args.reprod_id > 0 else "" 
    obj = "_".join(["gradcam", vis_args.MODEL.name, smooth_cond, random_cond])
    if reprod_id != "":
        obj += "_" + reprod_id

    # Get filenames and create absolute paths
    if os.path.isdir(vis_args.DATASET.path):
        files = os.listdir(vis_args.DATASET.path)
        paths = [os.path.join(vis_args.DATASET.path, f) for f in files]
    elif os.path.isfile(vis_args.DATASET.path):
        files = list(vis_args.DATASET.path.split("/")[-1])
        paths = [vis_args.DATASET.path]

    alpha = vis_args.RESULTS.alpha
    h = w = vis_args.DATASET.size

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
        #FIXME: Comment these 2 lines
        # if d == 100:
        #     break
        img = open_image(path)
        prep_img = preprocess_image(img, h=h, w=w)
        if args.cuda:
            prep_img = prep_img.to(device)

        cam, pred = gcv2.generate_cam(prep_img, vis_args.DATASET.target_class)
        cam_orig = cam.copy()

        # Run Pointing Game
        if vis_args.RESULTS.POINTING_GAME.state:
            cam = cam_orig
            prep_img = prep_img.cpu()
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
                                            obj=obj,
                                            alpha=alpha
                                           )
        if args.cuda:
            pred = pred.cpu()
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

    print('Grad cam completed')
