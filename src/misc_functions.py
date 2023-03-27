"""
Created on Thu Oct 21 11:09:09 2017

@author: Utku Ozbulak - github.com/utkuozbulak
@editor: Caner Ozer - github.com/canerozer
"""
import os
import copy
import math
from functools import reduce

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, UnidentifiedImageError, ImageDraw
import matplotlib.cm as mpl_color_map

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join('../results', file_name + '.jpg')
    save_image(gradient, path_to_file)


def custom_save_gradient_images(gradient, dst, file_name, obj=""):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if obj:
        #obj = "_" + obj
        dst = os.path.join(dst, obj)
        os.makedirs(dst, exist_ok=True)

    # Normalize
    gradient = gradient - gradient.min()
    gradient /= (gradient.max() - gradient.min())
    gradient = gradient
    # Save image
    file_name = ".".join(file_name.split(".")[:-1])
    path_to_file = os.path.join(dst, file_name + '.jpg')
    save_image(gradient, path_to_file)


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('../results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)


def custom_save_class_activation_images(org_img, activation_map, dst,
                                        file_name, obj="", alpha=0.4):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if obj != "":
        #obj = "_" + obj
        dst = os.path.join(dst, obj)
        os.makedirs(dst, exist_ok=True)

    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map,
                                                        'jet', alpha=alpha)
    
    # Save colored heatmap
    #path_to_file = os.path.join(dst, file_name + obj + ".png")
    #save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    file_name = ".".join(file_name.split(".")[:-1])
    path_to_file = os.path.join(dst, file_name + ".png")
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    #path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
    #save_image(activation_map, path_to_file)


def custom_save_np_arr(act_map, dst, file_name, obj=""):
    """
        Saves cam activation map as a numpy array

    Args:
        act_map (numpy arr): Activation map (grayscale) 0-1
        dst (str): Target path of the saving directory
        file_name (str): File name of the exported image
        obj (str): Objective, create a new folder if there exists
    """
    if obj:
        dst = os.path.join(dst, obj)
        os.makedirs(dst, exist_ok=True)

    file_name = ".".join(file_name.split(".")[:-1])
    path_to_file = os.path.join(dst, file_name + ".npz")
    np.save(path_to_file, act_map)


def apply_colormap_on_image(org_im, activation, colormap_name, alpha=0.4):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Threshold activation map?
    #activation = np.where(activation > 0.5, 0, activation)

    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)

    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    #heatmap[:, :, 3] = np.where(activation > 0.3, alpha, 0)
    heatmap[:, :, 3] = alpha
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray(((1-no_trans_heatmap)*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def preprocess_image(pil_im, resize_im=True, h=224, w=224):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((h, w), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)
    #recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


def get_example_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    example_list = (('../input_images/snake.jpg', 56),
                    ('../input_images/cat_dog.png', 243),
                    ('../input_images/spider.png', 72))
    img_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    # Read image
    original_image = Image.open(img_path).convert('RGB')
    # Process image
    prep_img = preprocess_image(original_image)
    # Define model
    pretrained_model = models.alexnet(pretrained=True)
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)


class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        elif isinstance(value, list):
            value = [DictAsMember(element)
                     if isinstance(element, dict)
                     else element
                     for element in value]

        return value
        

def get_model(model_meta):
    name = model_meta.name
    n_classes = model_meta.n_classes
    pretrained = model_meta.pretrained

    model = None
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    return model


def open_image(path):
    try:
        img = Image.open(path).convert('RGB')
    except UnidentifiedImageError:
        img = np.load(path)
        img = Image.fromarray(img.astype('float32')).convert('RGB')
    return img


def get_mask(f, src, size=None):
    try:
        mask = Image.open(os.path.join(src, f))
    except FileNotFoundError:
        return None

    mask = mask.resize(size, Image.NEAREST)
    mask = np.array(mask)
    mask[mask<127.] = 0
    mask[mask>127.] = 1
    return mask

def fit_bbox(mask):
    bb = []
    if mask is not None:
        idxs = np.where(mask == 1)
        if len(idxs[0]) > 0:
            xmin, xmax = idxs[1].min(), idxs[1].max() 
            ymin, ymax = idxs[0].min(), idxs[0].max() 
            bb.append([xmin, ymin, xmax, ymax])
    return bb


def get_boxes(gt_src, fn, img, size=(None, None)):
    df = pd.read_csv(gt_src)
    idx = df.loc[df.image_name == fn]
    gts = idx.iloc[0]["annotation"]

    w, h = np.array(img).shape[:2]

    bbs = []

    if type(gts) == str:
        gts = gts.split(";")
        for gt in gts:
            x = []
            y = []

            gt = gt[2:]
            gt = gt.split(" ")
            for i in range(len(gt)):
                if i % 2 == 0:
                    x.append(float(gt[i]))
                else:
                    y.append(float(gt[i]))
            xmin = min(x) / h * size[0]
            ymin = min(y) / w * size[1]
            xmax = max(x) / h * size[0]
            ymax = max(y) / w * size[1]
            bbs.append([xmin, ymin, xmax, ymax])

    return bbs


def show_bbox(img, cam, fn, bbox_gt_src=None, mask_gt_src=None, color="black",
              cross_max=True):

    if mask_gt_src:
        mask = get_mask(fn, mask_gt_src, size=cam.shape)
        bbs = fit_bbox(mask)
        # if mask is not None:
        #     idxs = np.where(mask > 0.5)
        #     cam[idxs] = 1
    else:
        bbs = get_boxes(bbox_gt_src, fn, img, size=np.array(img).shape)

    try:
        cam = Image.fromarray(cam)
    except TypeError:
        if len(cam.shape) == 3 and cam.shape[0] == 1:
            cam = Image.fromarray(cam[0])
    drawer = ImageDraw.Draw(cam)

    if cross_max:
        max_val = np.amax(cam)
        loc = np.where(max_val == cam)
        max_loc = tuple(zip(loc[0], loc[1]))[0]
        xmin, ymin = max_loc[1] - 4, max_loc[0] - 4
        xmax, ymax = max_loc[1] + 4, max_loc[0] + 4
        drawer.line((xmin, ymin, xmax, ymax), fill=0, width=4)
        drawer.line((xmax, ymin, xmin, ymax), fill=0, width=4)

    for bb in bbs:
        drawer.rectangle(bb, outline=color, width=2)

    cam = np.array(cam)
        
    return cam


def get_tensor_size(x):
    if isinstance(x, torch.Tensor):
        return x.element_size() * x.nelement() / 2**20
    elif isinstance(x, np.ndarray):
        return x.nbytes / 2**20


'''
def blur_input_array(array, kernel_size=11, sigma=5.):
    """Blur array with a 2D gaussian blur.

    Args:
        array: np.array, 3 or 4D tensor to blur.
        kernel_size: int, size of 2D kernel.
        sigma: float, standard deviation of gaussian kernel.

    Returns:
        4D np.array that has been smoothed with gaussian kernel.
    """
    ndim = len(array.shape)
    if ndim == 3:
        array = array[None]
    assert ndim == 4
    num_channels = array.shape[1]
    
    x_cord = np.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).reshape(kernel_size, kernel_size)
    y_grid = x_grid.T
    xy_grid = np.stack([x_grid, y_grid], axis=-1).float()

    mean = (kernel_size - 1) / 2
    variance = sigma**2

    gaussian_kernel = (1./(2.*math.pi*variance)) * np.exp(
        -1*np.sum((xy_grid - mean)**2., axis=-1) /
        (2.*variance)
    )

    gaussian_kernel /= np.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel[None, None]
'''

def blur_input_tensor(tensor, kernel_size=11, sigma=5.0):
    """Blur tensor with a 2D gaussian blur.

    Args:
        tensor: torch.Tensor, 3 or 4D tensor to blur.
        kernel_size: int, size of 2D kernel.
        sigma: float, standard deviation of gaussian kernel.

    Returns:
        4D torch.Tensor that has been smoothed with gaussian kernel.
    """
    ndim = len(tensor.shape)
    if ndim == 3:
        tensor = tensor.unsqueeze(0)
        ndim = len(tensor.shape)
    assert ndim == 4
    num_channels = tensor.shape[1]
    device = tensor.device

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(
        -1*torch.sum((xy_grid - mean)**2., dim=-1) /
        (2.*variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(num_channels, 1, 1, 1)
    gaussian_kernel = gaussian_kernel.to(device)

    padding = nn.ReflectionPad2d(int(mean)).to(device)
    gaussian_filter = nn.Conv2d(in_channels=num_channels,
                                out_channels=num_channels,
                                kernel_size=kernel_size,
                                groups=num_channels,
                                bias=False).to(device)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    smoothed_tensor = gaussian_filter(padding(tensor))

    return smoothed_tensor

def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)
