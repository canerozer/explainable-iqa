"""
Created on Wed Jun 19 17:12:04 2019

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import argparse
import yaml
import os
import tqdm

import torch

from src.backprop import VanillaBackprop
from src.misc_functions import (get_example_params, convert_to_grayscale,
                                preprocess_image, DictAsMember,
                                custom_save_gradient_images, recreate_image,
                                open_image, get_boxes)
from src.models import _get_classification_model as get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Vanilla Backprop')
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

    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model, vis_args.MODEL.name)

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

    for f, path in tqdm.tqdm(zip(files, paths)):
        img = open_image(path)
        prep_img = preprocess_image(img, h=h, w=w)
        vanilla_grads = VBP.generate_gradients(prep_img,
                                               vis_args.DATASET.target_class)
        #custom_save_gradient_images(vanilla_grads, vis_args.RESULTS.dir,
        #                            f, obj="vanillabp")
        grad_x_image = vanilla_grads[0] * prep_img.detach().numpy()[0]
        bw_inputxgrad = convert_to_grayscale(grad_x_image)

        if vis_args.RESULTS.POINTING_GAME.state:
            boxes = get_boxes(vis_args.RESULTS.DRAW_GT_BBOX.gt_src, f, img)
            hit = pg.evaluate(boxes, bw_inputxgrad.squeeze())
            _ = pg.accumulate(hit[0], 1)

        prep_img = prep_img[0].mean(dim=0, keepdim=True)
        r = alpha * bw_inputxgrad + (1 - alpha) * prep_img.detach().numpy()
        custom_save_gradient_images(bw_inputxgrad, vis_args.RESULTS.dir,
                                    f, obj="bw_gradximage")

    if vis_args.RESULTS.POINTING_GAME.state:
        print(pg.print_stats())

    print('Grad x Image completed')

