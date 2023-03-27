"""
Created on Wed Jun 19 17:12:04 2019

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import argparse
import yaml
import os
import tqdm
import pandas as pd

import torch

from src.misc_functions import (get_example_params, convert_to_grayscale,
                                preprocess_image, DictAsMember,
                                custom_save_np_arr,
                                custom_save_class_activation_images,
                                custom_save_gradient_images, recreate_image,
                                open_image, get_boxes, get_mask, fit_bbox,
                                blur_input_tensor)
from src.models import _get_classification_model as get_model
from custom_vanilla_backprop import VanillaBackprop


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Vanilla Backprop')
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

    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model, vis_args.MODEL.name, device=device)

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


        img = open_image(path)
        prep_img = preprocess_image(img, h=h, w=w)
        if args.cuda:
            prep_img = prep_img.cuda()
        vanilla_grads = VBP.generate_gradients(prep_img,
                                               vis_args.DATASET.target_class)

        if args.cuda:
            prep_img = prep_img.cpu()
        grad_x_image = vanilla_grads[0] * prep_img.detach().numpy()[0]
        bw_inputxgrad = convert_to_grayscale(grad_x_image)

        if vis_args.RESULTS.POINTING_GAME.state:
            if vis_args.RESULTS.POINTING_GAME.SMOOTHING.state:
                sigma = vis_args.RESULTS.POINTING_GAME.SMOOTHING.sigma
                bw_inputxgrad = torch.Tensor(bw_inputxgrad[None])
                bw_inputxgrad = blur_input_tensor(bw_inputxgrad, sigma=sigma).numpy()[0]
            if vis_args.DATASET.MASK.state:
                mask = get_mask(f, vis_args.DATASET.MASK.path, size=(h,w))
                boxes = fit_bbox(mask)
            elif vis_args.DATASET.BBOX.state:
                mask = None
                boxes = get_boxes(vis_args.DATASET.BBOX.path, f, img,
                                  size=(h,w))
            hit = pg.evaluate(bw_inputxgrad.squeeze(), mask=mask, boxes=boxes)
            _ = pg.accumulate(hit[0], 1)
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
        obj = "_".join(["bw_gradximage", vis_args.MODEL.name, smooth_cond, random_cond])
        if reprod_id != "":
            obj += "_" + reprod_id

        prep_img = prep_img[0].mean(dim=0, keepdim=True)
        if args.cuda:
            prep_img = prep_img.cpu()

        img = img.resize((h, w))
        custom_save_class_activation_images(img, bw_inputxgrad[0],
                                            vis_args.RESULTS.dir, f,
                                            obj=obj, alpha=alpha
        )

        d += 1


    if vis_args.RESULTS.POINTING_GAME.state:
        print(pg.print_stats())

    pg_img_names, pg_preds = list(pg_dict.keys()), list(pg_dict.values())
    pg_df = pd.DataFrame({"image_names": pg_img_names, "predictions": pg_preds})
    pg_df_path = os.path.abspath(os.path.join(vis_args.RESULTS.save_preds_to, '..'))
    pg_df.to_csv(os.path.join(pg_df_path, obj, 'pg.csv'))

    print('Grad x Image completed')

