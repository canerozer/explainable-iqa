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

import torch

from src.misc_functions import get_example_params, DictAsMember, \
                               open_image, preprocess_image, \
                               recreate_image, show_bbox, \
                               custom_save_class_activation_images, get_boxes
from src.models import _get_classification_model as get_model
from src.normgrad import NormGrad


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize NormGrad')
    parser.add_argument('--yaml', type=str, metavar="YAML",
                        default="configs/R50_LVOT.yaml",
                        help='Enter the path for the YAML config')
    parser.add_argument('--img', type=str, metavar='I', default=None,
                        help="Enter the image path")
    parser.add_argument('--target', type=int, metavar='T', default=None,
                        help="Enter the target class ID")
    args = parser.parse_args()

    yaml_path = args.yaml
    with open(yaml_path, 'r') as f:
        vis_args = DictAsMember(yaml.safe_load(f))

    if args.img:
        vis_args.DATASET.path = args.img
        vis_args.DATASET.target_class = args.target

    # Load model & pretrained params
    pretrained_model = get_model(vis_args.MODEL)
    state = torch.load(vis_args.MODEL.path)
    try:
        pretrained_model.load_state_dict(state["model"])
    except KeyError as e:
        pretrained_model.load_state_dict(state)

    alpha = vis_args.RESULTS.alpha
    h = w = vis_args.DATASET.size

    # Initialize NormGrad
    normgrad = NormGrad(pretrained_model, vis_args.MODEL,
                        input_size=(h, w))
    try:
        phase1 = vis_args.MODEL.phase1
    except KeyError:        
        phase1 = "scaling"

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
        tolerance = vis_args.RESULTS.POINTING_GAME.tolerance
        pg = PointingGame(vis_args.MODEL.n_classes, tolerance=tolerance)

    preds_dict = {}

    for f, path in tqdm.tqdm(zip(files, paths)):
        img = open_image(path)
        prep_img = preprocess_image(img, h=h, w=w)
        cam, pred = normgrad(prep_img, phase1, vis_args.DATASET.target_class)
        if vis_args.RESULTS.POINTING_GAME.state:
            boxes = get_boxes(vis_args.RESULTS.DRAW_GT_BBOX.gt_src, f, img)
            hit = pg.evaluate(boxes, cam)
            _ = pg.accumulate(hit[0], 1)
        if vis_args.RESULTS.DRAW_GT_BBOX.state:
            cam = show_bbox(img, cam, f, vis_args.RESULTS.DRAW_GT_BBOX.gt_src)
        img = img.resize((h, w))
        custom_save_class_activation_images(img, cam,
                                            vis_args.RESULTS.dir, f,
                                            obj="normgrad", alpha=alpha
                                           )
        preds_dict[f] = pred

    if vis_args.RESULTS.POINTING_GAME.state:
        print(pg.print_stats())
    img_names, preds = list(preds_dict.keys()), list(preds_dict.values())
    df = pd.DataFrame({"image_names": img_names, "predictions": preds})
    df.to_csv(vis_args.RESULTS.save_preds_to)
        
    print('NormGrad completed')
