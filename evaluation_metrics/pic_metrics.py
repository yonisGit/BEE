# Boilerplate imports.
# import tensorflow as tf
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import torch

# # From our repository.
# import saliency.core as saliency
# from saliency.metrics import pic
from saliency_master.saliency import core as saliency
from saliency_master.saliency.metrics import pic
from saliency_utils import *


# Boilerplate methods.
def show_image(im, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.imshow(im)
    ax.set_title(title)


def show_grayscale_image(im, title='', ax=None):
    if ax is None:
        plt.figure()
    plt.axis('off')

    plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.title(title)


def show_curve_xy(x, y, title='PIC', label=None, color='blue',
                  ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    auc = np.trapz(y) / y.size
    label = f'{label}, AUC={auc:.3f}'
    ax.plot(x, y, label=label, color=color)
    ax.set_title(title)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend()


def show_curve(compute_pic_metric_result, title='PIC', label=None, color='blue',
               ax=None):
    show_curve_xy(compute_pic_metric_result.curve_x,
                  compute_pic_metric_result.curve_y, title=title, label=label,
                  color=color,
                  ax=ax)


def show_blurred_images_with_scores(compute_pic_metric_result):
    # Get model prediction scores.
    images_to_display = compute_pic_metric_result.blurred_images
    scores = compute_pic_metric_result.predictions
    thresholds = compute_pic_metric_result.thresholds

    # Visualize blurred images.
    nrows = (len(images_to_display) - 1) // 5 + 1
    fig, ax = plt.subplots(nrows=nrows, ncols=5,
                           figsize=(20, 20 / 5 * nrows))
    for i in range(len(images_to_display)):
        row = i // 5
        col = i % 5
        title = f'score: {scores[i]:.3f}\nthreshold: {thresholds[i]:.3f}'
        show_image(images_to_display[i], title=title, ax=ax[row, col])


def load_image(file_path):
    im = PIL.Image.open(file_path)
    im = im.resize((224, 224))
    im = np.asarray(im)
    return im


import torchvision.transforms as transforms


def preprocess_image(im, size=224):
    # im = tf.keras.applications.vgg16.preprocess_input(im)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # if is_transformer:
    #     normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform1 = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        # normalize,
    ])

    im = transform1(PIL.Image.fromarray(im[0]))

    return im


# Defines the fractions (thresholds) of top salient pixels for which the intermediate
# blurred images should be created and evaluated by the model. The higher the number of
# different thresholds is, the more accurate the PIC curve is.
saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13,
                       0.21, 0.34, 0.5, 0.75]


# Saliency map to use for the metrics. Here we use the absolute value
# of the sum of attributions over the color channel.

# gig_saliency_map = np.abs(np.sum(guided_ig_mask_3d, axis=2))


# Define prediction function.
def create_predict_function_softmax(model, class_idx):
    """Creates the model prediction function that can be passed to compute_pic_metric method.

      The function returns the softmax value for the Softmax Information Curve.
    Args:
      class_idx: the index of the class for which the model prediction should
        be returned.
    """

    def predict(image_batch):
        """Returns model prediction for a batch of images.

        The method receives a batch of images in uint8 format. The method is responsible to
        convert the batch to whatever format required by the model. In this particular
        implementation the conversion is achieved by calling preprocess_input().

        Args:
          image_batch: batch of images of dimension [B, H, W, C].

        Returns:
          Predictions of dimension [B].
        """
        # image_batch = tf.keras.applications.vgg16.preprocess_input(image_batch)
        image_batch = preprocess_image(image_batch)

        # plt.imshow(image_batch.permute(1, 2, 0))

        score = model(image_batch.unsqueeze(0).cuda())[:, class_idx]
        return score.detach().cpu().numpy()

    return predict



# Define prediction function.
def create_predict_function_accuracy(model, class_idx):
    """Creates the model prediction function that can be passed to compute_pic_metric method.

      The function returns the accuracy for the Accuracy Information Curve.

    Args:
      class_idx: the index of the class for which the model prediction score should
        be returned.
    """

    def predict(image_batch):
        """Returns model accuracy for a batch of images.

        The method receives a batch of images in uint8 format. The method is responsible to
        convert the batch to whatever format required by the model. In this particular
        implementation the conversion is achieved by calling preprocess_input().

        Args:
          image_batch: batch of images of dimension [B, H, W, C].

        Returns:
          Predictions of dimension [B], where every element is either 1.0 for correct
          prediction or 0.0 for incorrect prediction.
        """
        image_batch = preprocess_image(image_batch)
        scores = model(image_batch.unsqueeze(0).cuda())
        arg_max = np.argmax(scores, axis=1)
        accuracy = arg_max == class_idx
        return np.ones_like(arg_max) * accuracy.detach().cpu().numpy()

    return predict


def get_pred_func_sic(model, prediction_class):
    pred_func_sic = create_predict_function_softmax(model, prediction_class)
    return pred_func_sic


def get_pred_func_aic(model, prediction_class):
    pred_func_sic = create_predict_function_accuracy(model, prediction_class)
    return pred_func_sic


def calculate_sic(model, gig_saliency_map, im_orig, label):
    # Compute PIC for Guided IG.
    try:
        # Create a random mask for the initial fully blurred image.
        random_mask = pic.generate_random_mask(image_height=im_orig.shape[0],
                                               image_width=im_orig.shape[1],
                                               fraction=0.01)
        pred_func_sic = get_pred_func_sic(model, label)
        # saliency_thresholds = [0.01, 0.02, 0.03, 0.05]
        # saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13,
        #                        0.21, 0.34, 0.5, 0.75]
        saliency_thresholds = [0.005, 0.01, 0.03, 0.08, 0.10,
                               0.21, 0.34, 0.5, 0.75]
        gig_result_sic = pic.compute_pic_metric(img=im_orig,
                                                saliency_map=gig_saliency_map,
                                                random_mask=random_mask,
                                                pred_func=pred_func_sic,
                                                # min_pred_value=-1000,
                                                min_pred_value=0.5,
                                                saliency_thresholds=saliency_thresholds,
                                                keep_monotonous=True,
                                                # num_data_points=100)
                                                num_data_points=1000)
    except pic.ComputePicMetricError as e:
        # In normal circumstances skip the image. Here, we re-raise the error.
        raise e

    # Don't forget to check for the None result.
    if gig_result_sic is None:
        raise AssertionError(
            "The fully blurred image has same or higher score as compared to the original image.")
    else:
        return gig_result_sic


def calculate_aic(model, gig_saliency_map, im_orig, label):
    # Compute PIC for Guided IG.
    try:
        # Create a random mask for the initial fully blurred image.
        random_mask = pic.generate_random_mask(image_height=im_orig.shape[0],
                                               image_width=im_orig.shape[1],
                                               fraction=0.01)
        pred_func_sic = get_pred_func_aic(model, label)
        # saliency_thresholds = [0.01, 0.02, 0.03, 0.05]
        saliency_thresholds = [0.005, 0.01, 0.03, 0.08, 0.10,
                               0.21, 0.34, 0.5, 0.75]
        gig_result_aic = pic.compute_pic_metric(img=im_orig,
                                                saliency_map=gig_saliency_map,
                                                random_mask=random_mask,
                                                pred_func=pred_func_sic,
                                                # min_pred_value=-1000,
                                                min_pred_value=0.5,
                                                saliency_thresholds=saliency_thresholds,
                                                keep_monotonous=True,
                                                # num_data_points=100)
                                                num_data_points=1000)
    except pic.ComputePicMetricError as e:
        # In normal circumstances skip the image. Here, we re-raise the error.
        raise e

    # Don't forget to check for the None result.
    if gig_result_aic is None:
        raise AssertionError(
            "The fully blurred image has same or higher score as compared to the original image.")
    else:
        return gig_result_aic


def compare_two_methods(saliency_map1, saliency_map2, im_orig, label):
    fig, ax = plt.subplots(figsize=(12, 6))
    title = "PIC - Softmax Information Curve"
    sic1 = calculate_sic(saliency_map1, im_orig, label)
    sic2 = calculate_sic(saliency_map2, im_orig, label)
    show_curve(sic1, title=title, label='Guided IG', color='blue', ax=ax)
    show_curve(sic2, title=title, label='Random', color='red', ax=ax)
