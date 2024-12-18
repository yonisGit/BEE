import os

import captum.attr
import numpy
import numpy as np
import pandas as pd
import torch.multiprocessing
import torchvision.models
from PIL import Image
from tqdm import tqdm
import random
from sklearn.metrics import auc
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from additional.imagenet_lables import label_map
from utils import *
from models import *
from pytorch_grad_cam.fullgrad_cam import FullGrad
from pytorch_grad_cam.layer_cam import LayerCAM
from pytorch_grad_cam.ablation_cam import AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from evaluation_metrics import evaluations

BEE = 'bee'

IMAGE_SIZE = 'image_size'

BBOX = 'bbox'

ROOT_IMAGES = "{0}/imgs/ILSVRC2012_img_val"
IS_VOC = False
IS_COCO = False
IS_VOC_BBOX = False
IS_COCO_BBOX = False
INPUT_SCORE = 'score_original_image'
IMAGE_PIXELS_COUNT = 50176
INTERPOLATION_STEPS = 4
LABEL = 'label'
TOP_K_PERCENTAGE = 0.25
USE_TOP_K = False
BY_MAX_CLASS = True
GRADUAL_PERTURBATION = True
IS_TRAIN = True
IMAGE_PATH = 'image_path'
DATA_VAL_TXT = 'imgs/val.txt'
TIMESTEPS = 500
device = 'cuda'

NUMBER_OF_IMAGES_TO_TRAIN_ON = 5000
NUMBER_OF_EPOCHS_TESTING = 100
NUMBER_OF_EPOCHS_TRAINING = 100
HOW_MUCH_BASELINES_IMAGES = 100
TIMES_TO_REPEAT_TEST = 100
TIMES_TO_REPEAT = 1

NORMAL = 'NORMAL'
UNIFORM = 'UNIFORM'
BLUR = 'BLUR'
CONST = 'CONST'
TRAIN_DATA = 'TRAIN'
PIC = 'PIC'
ADP = 'ADP'
NEG = 'NEG'
POS = 'POS'
INS = 'INS'
DEL = 'DEL'
AIC = 'AIC'
SIC = 'SIC'
METRICS = [PIC, ADP, NEG, POS, INS, DEL, AIC, SIC]
TYPES = [NORMAL, UNIFORM, BLUR, CONST, TRAIN_DATA]
HIGHER_LOWER_MAP = {PIC: 0, ADP: 1, AIC: 0, SIC: 0, NEG: 0, POS: 1, INS: 0, DEL: 1}

metric_distributions = {
    'PIC': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
    'ADP': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
    'AIC': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
    'SIC': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
    'NEG': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
    'POS': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
    'INS': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
    'DEL': {'trials': [0, 0, 0], 'wins': [0, 0, 0]}
}

metric_best_per_sample = {
    'PIC': -1,
    'ADP': 1000.0,
    'AIC': -1,
    'SIC': -1,
    'NEG': -1,
    'POS': 1000.0,
    'INS': -1,
    'DEL': 1000.0,
}

family_bests = {
    'NORMAL': metric_best_per_sample.copy(),
    'UNIFORM': metric_best_per_sample.copy(),
    'BLUR': metric_best_per_sample.copy(),
    'TRAIN': metric_best_per_sample.copy(),
    'CONST': metric_best_per_sample.copy(),
}

chosen_bests = {
    'CHOSEN': metric_best_per_sample.copy(),
    'UNIFORM': metric_best_per_sample.copy(),
}

metric_history = {
    'PIC': [],
    'ADP': [],
    'AIC': [],
    'SIC': [],
    'NEG': [],
    'POS': [],
    'INS': [],
    'DEL': [],
}

hundred_zeros = torch.zeros((100,))
hundred_ones = torch.ones((100,))
means = [hundred_zeros.clone(), hundred_zeros.clone(), hundred_zeros.clone()]
variances = [hundred_ones.clone(), hundred_ones.clone(), hundred_ones.clone()]


def get_grads_wrt_image(model, label, images_batch, device='cuda', steps=50):
    model.eval()
    model.zero_grad()

    images_batch.requires_grad = True
    preds = model(images_batch.to(device), hook=True)
    _, predicted = torch.max(preds.data, 1)
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()
    with torch.no_grad():
        image_grads = images_batch.grad.detach()
    images_batch.requires_grad = False
    return image_grads


def backward_class_score_and_get_activation_grads(model, label, x, only_post_features=False, device='cuda',
                                                  is_middle=False):
    model.zero_grad()

    preds = model(x.to(device), hook=True, only_post_features=only_post_features,
                  is_middle=is_middle)
    _, predicted = torch.max(preds.data, 1)
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()

    activations_gradients = model.get_activations_gradient().unsqueeze(
        1).detach().cpu()

    return activations_gradients


def backward_class_score_and_get_images_grads(model, label, x, only_post_features=False, device='cuda'):
    model.zero_grad()
    preds = model(x.squeeze(1).to(device), hook=True)
    _, predicted = torch.max(preds.data, 1)
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()

    images_gradients = model.get_activations_gradient().unsqueeze(
        1).detach().cpu()

    return images_gradients


def get_blurred_values(target, num_steps):
    num_steps += 1
    if num_steps <= 0: return np.array([])
    target = target.squeeze()
    tshape = len(target.shape)
    blurred_images_list = []
    for step in range(num_steps):
        sigma = int(step) / int(num_steps)
        sigma_list = [sigma, sigma, 0]

        if tshape == 4:
            sigma_list = [sigma, sigma, sigma, 0]

        blurred_image = ndimage.gaussian_filter(
            target.detach().cpu().numpy(), sigma=sigma_list, mode="grid-constant")
        blurred_images_list.append(blurred_image)

    return numpy.array(blurred_images_list)


def get_images(image_path, interpolation_on_images_steps):
    CWD = os.getcwd()
    root = ROOT_IMAGES.format(CWD)

    print(image_path)

    img = Image.open(root + '/' + image_path).convert('RGB')
    im = preprocess(img)
    X = torch.stack([im])

    if interpolation_on_images_steps > 0:
        X = get_interpolated_values(torch.zeros_like(im), im, num_steps=interpolation_on_images_steps)

    return X


def gaussian_blur(image, sigma):
    """Returns Gaussian blur filtered 3d (WxHxC) image.

    Args:
      image: 3 dimensional ndarray / input image (W x H x C).
      sigma: Standard deviation for Gaussian blur kernel.
    """
    if sigma == 0:
        return image
    return gaussian_filter(
        image, sigma=[sigma, sigma, 0], mode="constant")


def get_images_blur(image_path, interpolation_on_images_steps):
    CWD = os.getcwd()
    root = ROOT_IMAGES.format(CWD)

    print(image_path)

    img = Image.open(root + '/' + image_path).convert('RGB')
    im = preprocess(img)
    X = torch.stack([im])

    if interpolation_on_images_steps > 0:
        X = torch.tensor(get_blurred_values(im.detach(),
                                            interpolation_on_activations_steps_arr[-1]))

    return X


def apply_contextual_thompson(input, model, max_index, w, metric_score):
    y = metric_score.to(device)
    m = means[max_index].clone()
    variance = variances[max_index].clone()
    q = (1.0 / variance).to(device)

    objective = (lambda w_i, m_i, q_i, x_i: (0.5 * (torch.sum(q_i * ((w_i - m_i) ** 2))) + torch.log(
        1 + torch.exp(-y * w_i * x_i))))

    params = list(model.parameters()) + list([w])

    optimizer = torch.optim.AdamW(params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)

    target = torch.rand((1, 100,)).to(device)
    number_of_epochs = NUMBER_OF_EPOCHS_TRAINING
    for epoch in range(number_of_epochs):
        optimizer.zero_grad()
        input_features = model(input.cuda())

        res = objective(w, m.cuda(), q.cuda(), input_features)
        loss = torch.sum(res)

        if loss == None or torch.isnan(loss):
            break
        print(f'loss : {loss}, epoch: {epoch}')

        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step(loss)

        if scheduler.last_epoch > 10 and scheduler.best - 0.001 < loss:
            break

    p = 1.0 / (1.0 + torch.exp(-w * input_features)).detach()
    q = q.detach() + torch.sum((input_features ** 2) * p * (1 - p)).detach()
    means[max_index] = w.detach()
    variances[max_index] = (1.0 / q).detach()


def get_by_class_saliency_bee(image_path,
                              label,
                              operations,
                              model_name='densnet',
                              layers=[12],
                              interpolation_on_images_steps_arr=[0, 50],
                              interpolation_on_activations_steps_arr=[0, 50],
                              device='cuda',
                              use_mask=False, mode='normal'):
    if mode == TYPES[0]:
        images, integrated_heatmaps = heatmap_of_layer_bee_normal(device, image_path, label,
                                                                  layers,
                                                                  model_name)
    elif mode == TYPES[1]:
        images, integrated_heatmaps = heatmap_of_layer_bee_uniform(device, image_path, label,
                                                                   layers,
                                                                   model_name)
    elif mode == TYPES[2]:
        images, integrated_heatmaps = heatmap_of_layer_bee_blur(device, image_path, label,
                                                                layers,
                                                                model_name)
    elif mode == TYPES[3]:
        images, integrated_heatmaps = heatmap_of_layer_bee_constant(device, image_path, label,
                                                                    layers,
                                                                    model_name)
    else:
        images, integrated_heatmaps = heatmap_of_layer_activations_integration(device, image_path,
                                                                               interpolation_on_activations_steps_arr,
                                                                               interpolation_on_images_steps_arr,
                                                                               label,
                                                                               layers,
                                                                               model_name)

    heatmap = make_resize_norm(integrated_heatmaps)

    last_image = images[-1]

    t = tensor2cv(last_image)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, im, heatmap_cv, blended_img_mask, last_image, score, heatmap


def get_by_class_saliency_bee_traindata(image_path,
                                        label,
                                        operations,
                                        model_name='densnet',
                                        layers=[12],
                                        interpolation_on_images_steps_arr=[0, 50],
                                        interpolation_on_activations_steps_arr=[0, 50],
                                        device='cuda',
                                        use_mask=False, image_paths={}):
    images, integrated_heatmaps = heatmap_of_layer_bee_traindata(device, image_path, label,
                                                                 layers,
                                                                 model_name, image_paths)

    heatmap = make_resize_norm(integrated_heatmaps)

    last_image = images[-1]

    t = tensor2cv(last_image)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, im, heatmap_cv, blended_img_mask, last_image, score, heatmap


def make_resize_norm(act_grads):
    heatmap = torch.sum(act_grads.squeeze(0), dim=0)
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)

    heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().cpu().data.numpy()
    return heatmap


def heatmap_of_layer_activations_integration_mulacts(device, image_path, interpolation_on_activations_steps_arr,
                                                     interpolation_on_images_steps_arr,
                                                     label, layers, model_name):
    print(layers[0])
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    images = get_images(image_path, interpolation_on_images_steps=0)
    label = torch.tensor(label, dtype=torch.long, device=device)
    activations = model.get_activations(images.to(device)).cpu()

    activations_featmap_list = (activations.unsqueeze(1))

    x, _ = torch.min(activations_featmap_list, dim=1)
    basel = torch.ones_like(activations_featmap_list) * x.unsqueeze(1)
    igacts = get_interpolated_values(basel.detach(), activations_featmap_list,
                                     INTERPOLATION_STEPS).detach()

    grads = []
    for act in igacts:
        act.requires_grad = True

        diff2 = (act - basel) / INTERPOLATION_STEPS
        normalic2 = torch.norm(diff2)

        grads.append(F.relu(calc_grads_model(model, act, device, label).detach()) * F.relu(act))
        act = act.detach()
        act.requires_grad = False

    with torch.no_grad():
        igrads = torch.stack(grads).detach()

        mul_grad_act = (igrads.squeeze().detach())
        gradsum = torch.sum(mul_grad_act, dim=[0])
        integrated_heatmaps = gradsum

    return images, integrated_heatmaps


def heatmap_of_layer_activations_integration(device, image_path, interpolation_on_activations_steps_arr,
                                             interpolation_on_images_steps_arr,
                                             label, layers, model_name):
    print(layers[0])
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    images = get_images(image_path, interpolation_on_images_steps=0)
    label = torch.tensor(label, dtype=torch.long, device=device)
    activations = model.get_activations(images.to(device)).cpu()

    activations_featmap_list = (activations.unsqueeze(1))

    x, _ = torch.min(activations_featmap_list, dim=0)
    basel = torch.ones_like(activations_featmap_list) * x.unsqueeze(0)

    igacts = get_interpolated_values(basel.detach(), activations_featmap_list,
                                     INTERPOLATION_STEPS).detach()

    grads = []
    for act in igacts:
        act.requires_grad = True

        diff2 = (act - basel) / INTERPOLATION_STEPS
        normalic2 = torch.norm(diff2)

        grads.append(calc_grads_model(model, act, device, label).detach() * act)
        act = act.detach()
        act.requires_grad = False

    with torch.no_grad():
        igrads = torch.stack(grads).detach()
        mul_grad_act = F.relu(igrads.squeeze().detach())
        integrated_heatmaps = torch.sum(mul_grad_act, dim=[0])

    return images, integrated_heatmaps


def heatmap_of_layer_before_last_integrand(device, image_path, interpolation_on_activations_steps_arr,
                                           interpolation_on_images_steps_arr,
                                           label, layers, model_name):
    print(layers)
    model = GradModel(model_name, feature_layer=layers)
    model.to(device)
    model.eval()
    model.zero_grad()

    images = get_images(image_path, interpolation_on_images_steps=0)

    label = torch.tensor(label, dtype=torch.long, device=device)

    original_activations = model.get_activations(images.to(device)).cpu()
    original_activations_featmap_list = original_activations

    x, _ = torch.min(original_activations_featmap_list, dim=1)
    basel = torch.ones_like(original_activations_featmap_list) * x.unsqueeze(1)

    igacts = get_interpolated_values(basel.detach(), original_activations_featmap_list,
                                     INTERPOLATION_STEPS).detach()

    grads = []
    for act in igacts:
        act.requires_grad = True

        diff2 = (act - basel) / INTERPOLATION_STEPS
        normalic2 = torch.norm(diff2)
        grads.append((calc_grads_model(model, act, device, label).detach()) * F.relu(act) * normalic2)
        act = act.detach()
        act.requires_grad = False

    with torch.no_grad():
        igrads = torch.stack(grads).detach()
        mul_grad_act = F.relu(igrads.squeeze().detach())
        integrated_heatmaps = torch.sum(mul_grad_act, dim=[0])

    return images, integrated_heatmaps


def heatmap_of_layers_layer_no_interpolation(device, image_path, interpolation_on_activations_steps_arr,
                                             interpolation_on_images_steps_arr,
                                             label, layers, model_name):
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    images = get_images(image_path, interpolation_on_images_steps=0)

    label = torch.tensor(label, dtype=torch.long, device=device)
    activations = model.get_activations(images.to(device)).cpu()
    activations_featmap_list = (activations.unsqueeze(1))
    gradients = calc_grads_model(model, activations_featmap_list, device, label).detach()
    gradients_squeeze = gradients.detach().squeeze()
    act_grads = F.relu(activations.squeeze()) * F.relu(gradients_squeeze) ** 2
    integrated_heatmaps = torch.sum(act_grads.squeeze(0), dim=0).unsqueeze(0).unsqueeze(0)
    return images, integrated_heatmaps


def heatmap_of_layer_bee_normal(device, image_path, label, layers, model_name):
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    images = get_images(image_path, interpolation_on_images_steps=0)
    label = torch.tensor(label, dtype=torch.long, device=device)
    activations = model.get_activations(images.to(device)).cpu()

    full_tensor = get_noise_values_bee(model, activations, label)

    with torch.no_grad():
        mul_grad_act = (full_tensor.squeeze().detach())
        integrated_heatmaps = torch.sum(mul_grad_act, dim=[0])

    return images, integrated_heatmaps


def heatmap_of_layer_bee_uniform(device, image_path, label, layers, model_name):
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    images = get_images(image_path, interpolation_on_images_steps=0)
    label = torch.tensor(label, dtype=torch.long, device=device)
    activations = model.get_activations(images.to(device)).cpu()

    full_tensor = get_noise_values_bee(model, activations, label, mode=UNIFORM)

    with torch.no_grad():
        mul_grad_act = (full_tensor.squeeze().detach())
        integrated_heatmaps = torch.sum(mul_grad_act, dim=[0])

    return images, integrated_heatmaps


def heatmap_of_layer_bee_blur(device, image_path, label, layers, model_name):
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    images = get_images(image_path, interpolation_on_images_steps=0)
    label = torch.tensor(label, dtype=torch.long, device=device)
    activation = model.get_activations(images.to(device)).cpu()
    activations_featmap_list = (activation.unsqueeze(1))
    max_sigma = random.randint(3, 50)

    basel = gaussian_blur(activation, max_sigma)
    igacts = get_interpolated_values(basel.detach(), activations_featmap_list,
                                     INTERPOLATION_STEPS).detach()
    grads = []
    for act in igacts:
        act.requires_grad = True
        grads.append(F.relu(calc_grads_model(model, act, device, label).detach()) * F.relu(act))
        act = act.detach()
        act.requires_grad = False

    with torch.no_grad():
        igrads = torch.stack(grads).detach()
        mul_grad_act = (igrads.squeeze().detach())
        integrated_heatmaps = torch.sum(mul_grad_act, dim=[0])

    return images, integrated_heatmaps


def heatmap_of_layer_bee_traindata(device, image_path, label, layers, model_name, images_df):
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    input_image = get_images(image_path, interpolation_on_images_steps=0)
    label = torch.tensor(label, dtype=torch.long, device=device)
    activations = model.get_activations(input_image.to(device)).cpu()
    activations_featmap_list = (activations.unsqueeze(1))

    baselines_paths = []
    number_of_baselines = HOW_MUCH_BASELINES_IMAGES
    for i in range(number_of_baselines):
        baselines_paths.append(images_df.sample()[IMAGE_PATH].item())

    eg_results = []
    for baseline_path in baselines_paths:
        baseline_image = get_images(baseline_path, interpolation_on_images_steps=0)
        basel = model.get_activations(baseline_image.to(device)).cpu()

        igacts = get_interpolated_values(basel.detach(), activations_featmap_list,
                                         INTERPOLATION_STEPS).detach()
        grads = []
        for act in igacts:
            act.requires_grad = True
            grads.append(F.relu(calc_grads_model(model, act, device, label).detach()) * F.relu(act))
            act = act.detach()
            act.requires_grad = False

        with torch.no_grad():
            igrads = torch.stack(grads).detach()
            mul_grad_act = (igrads.squeeze().detach())
            integrated_heatmaps_baseline = torch.sum(mul_grad_act, dim=[0])
        eg_results.append(torch.tensor(integrated_heatmaps_baseline))
    eg_results_tensor = torch.stack(eg_results)
    eg_results_tensor = torch.mean(eg_results_tensor, dim=[0, 1])

    with torch.no_grad():
        integrated_heatmaps = eg_results_tensor.squeeze()

    return input_image, integrated_heatmaps


def heatmap_of_layer_bee_constant(device, image_path, label, layers, model_name):
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    images = get_images(image_path, interpolation_on_images_steps=0)
    label = torch.tensor(label, dtype=torch.long, device=device)
    activations = model.get_activations(images.to(device)).cpu()
    baseline = sample_constant_per_channel(activations)

    activations_featmap_list = (activations.unsqueeze(1))
    igacts = get_interpolated_values(baseline.detach(), activations_featmap_list,
                                     INTERPOLATION_STEPS).detach()

    grads = []
    for act in igacts:
        act.requires_grad = True

        grads.append(F.relu(calc_grads_model(model, act, device, label).detach()) * F.relu(act))
        act = act.detach()
        act.requires_grad = False

    with torch.no_grad():
        igrads = torch.stack(grads).detach()
        mul_grad_act = (igrads.squeeze().detach())
        integrated_heatmaps = torch.sum(mul_grad_act, dim=[0, 1])

    return images, integrated_heatmaps


def get_noise_values_bee(model, activation, label, device='cuda', mode=NORMAL):
    ab_t = get_alphas_from_timestamp(TIMESTEPS).cuda()
    noise_tensor = torch.randn_like(activation).cuda()
    if mode == UNIFORM:
        noise_tensor = torch.rand_like(activation).cuda()
    perturb_activation = perturb_input(activation, TIMESTEPS, noise_tensor, ab_t)

    interpolated_activations = bee_baseline(model, label, device, activation, perturb_activation)

    return interpolated_activations


def bee_baseline(model, label, device, activations, pert_activation_last):
    minim = torch.amin(pert_activation_last.squeeze(), dim=[1, 2]).unsqueeze(1).unsqueeze(1)
    maxim = torch.amax(pert_activation_last.squeeze(), dim=[1, 2]).unsqueeze(1).unsqueeze(1)
    pert_activation_last = pert_activation_last - minim
    pert_activation_last /= maxim
    pert_activation_last = (pert_activation_last * (maxim - minim) + minim)

    igacts = get_interpolated_values(pert_activation_last.cpu(), activations.cpu(),
                                     INTERPOLATION_STEPS - 1).detach()

    grads = []
    prev = 0
    for act in igacts:
        act.requires_grad = True
        gradients_ig = calc_grads_model(model, act, device, label).detach()

        mul_gr_act = (gradients_ig).cuda() * (act).cuda()
        if prev != 0:
            mul_gr_act = (gradients_ig) * (act) * (act - prev) / INTERPOLATION_STEPS
            act.requires_grad = False
            prev = act
        grads.append((mul_gr_act))
        act = act.detach()
        act.requires_grad = False

    with torch.no_grad():
        igrads = torch.stack(grads).detach()
        return igrads


def sample_constant_per_channel(input_tensor):
    min_values, _ = torch.min(input_tensor, dim=2, keepdim=True)
    max_values, _ = torch.max(input_tensor, dim=2, keepdim=True)
    sampled_values = torch.rand(1, input_tensor.size(1), 1, 1) * (max_values - min_values) + min_values
    sampled_values = sampled_values.expand_as(input_tensor)

    return sampled_values


def get_alphas_from_timestamp(tsteps):
    beta1 = 1e-4
    beta2 = 0.02

    b_t = (beta2 - beta1) * torch.linspace(0, 1, tsteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1
    return ab_t


def perturb_input(x, step, noise, ab_t):
    return (ab_t.sqrt()[step, None, None, None] * x.cuda() + (1 - ab_t[step, None, None, None]) * noise).cuda()


def calc_grads_model(model, activations_featmap_list, device, label):
    activations_gradients = backward_class_score_and_get_activation_grads(model, label, activations_featmap_list,
                                                                          only_post_features=True,
                                                                          device=device)
    return activations_gradients


# LIFT-CAM
from captum.attr import DeepLift


class Model_Part(nn.Module):
    def __init__(self, model):
        super(Model_Part, self).__init__()
        self.model_type = None
        if model.model_str == 'convnext':
            self.avg_pool = model.avgpool
            self.classifier = model.classifier[-1]
        else:
            self.avg_pool = model.avgpool
            self.classifier = model.classifier

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def lift_cam(model, image_path, label, device, use_mask):
    images = get_images(image_path, 0)

    model.eval()
    model.zero_grad()
    output = model(images.to(device), hook=True)

    class_id = label
    if class_id is None:
        class_id = torch.argmax(output, dim=1)

    act_map = model.get_activations(images.to(device))

    model_part = Model_Part(model)
    model_part.eval()
    dl = DeepLift(model_part)
    ref_map = torch.zeros_like(act_map).to(device)
    dl_contributions = dl.attribute(act_map, ref_map, target=class_id, return_convergence_delta=False).detach()

    scores_temp = torch.sum(dl_contributions, (2, 3), keepdim=False).detach()
    scores = torch.squeeze(scores_temp, 0)
    scores = scores.cpu()

    vis_ex_map = (scores[None, :, None, None] * act_map.cpu()).sum(dim=1, keepdim=True)
    vis_ex_map = F.relu(vis_ex_map).float()

    with torch.no_grad():
        heatmap = vis_ex_map
        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = heatmap.squeeze().cpu().data.numpy()
        t = tensor2cv(images[-1])
        blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                           use_mask=use_mask)
    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def ig(model, image_path, label, device, use_mask):
    return saliency.ig(model, image_path, label, device, use_mask)


def blurig(model, image_path, label, device, use_mask):
    return saliency.blurig(model, image_path, label, device, use_mask)


def guidedig(model, image_path, label, device, use_mask):
    return saliency.guidedig(model, image_path, label, device, use_mask)


def ig_captum(model, image_path, label, device, use_mask):
    images = get_images(image_path, 0)

    model.eval()
    model.zero_grad()
    class_id = label

    integrated_grads = captum.attr.IntegratedGradients(model)
    baseline = torch.zeros_like(images).to(device)
    attr = integrated_grads.attribute(images.to(device), baseline, class_id)

    with torch.no_grad():
        heatmap = torch.mean(attr, dim=1, keepdim=True)
        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = heatmap.squeeze().cpu().data.numpy()
        t = tensor2cv(images[-1])
        blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                           use_mask=use_mask)
    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def get_torchgc_model_layer(network_name, device):
    if network_name.__contains__('resnet'):
        resnet101 = torchvision.models.resnet101(pretrained=True).to(device)
        resnet101_layer = resnet101.layer4
        return resnet101, resnet101_layer
    elif network_name.__contains__('convnext'):
        convnext = torchvision.models.convnext_base(pretrained=True).to(device)
        convnext_layer = convnext.features[-1]
        return convnext, convnext_layer

    densnet201 = torchvision.models.densenet201(pretrained=True).to(device)
    densnet201_layer = densnet201.features
    return densnet201, densnet201_layer


def ablation_cam_torchcam(network_name, image_path, label, device, use_mask):
    model, layer = get_torchgc_model_layer(network_name, device)
    cam_extractor = AblationCAM(model.to(device), layer)
    images = get_images(image_path, 0)
    targets = [ClassifierOutputTarget(label)]
    hm = cam_extractor(images.to(device), targets)

    heatmap = torch.tensor(hm).unsqueeze(0)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().detach().cpu().data.numpy()

    t = tensor2cv(images[-1])
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                       use_mask=use_mask)

    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def fullgrad_torchcam(network_name, image_path, label, device, use_mask):
    model, layer = get_torchgc_model_layer(network_name, device)
    cam_extractor = FullGrad(model, layer)
    images = get_images(image_path, 0)
    targets = [ClassifierOutputTarget(label)]
    hm = cam_extractor(images.to(device), targets)

    heatmap = torch.tensor(hm).unsqueeze(0)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().detach().cpu().data.numpy()

    t = tensor2cv(images[-1])
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                       use_mask=use_mask)

    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def layercam_torchcam(network_name, image_path, label, device, use_mask):
    model, layer = get_torchgc_model_layer(network_name, device)
    cam_extractor = LayerCAM(model.to(device), layer)
    images = get_images(image_path, 0)
    targets = [ClassifierOutputTarget(label)]
    hm = cam_extractor(images.to(device), targets)

    heatmap = torch.tensor(hm).unsqueeze(0)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().detach().cpu().data.numpy()

    t = tensor2cv(images[-1])
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                       use_mask=use_mask)

    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def run_all_operations(model, image_path, label, model_name='densenet', device='cpu', features_layer=8,
                       operations=[BEE],
                       use_mask=False):
    results = []
    for operation in operations:
        t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap = run_by_class_grad(model, image_path, label,
                                                                                              model_name,
                                                                                              device,
                                                                                              features_layer,
                                                                                              operation, use_mask)
        results.append((t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap))
    return results


def run_by_class_grad(model, image_path, label, model_name='densenet', device='cpu', features_layer=8, operation='ours',
                      use_mask=False):
    CWD = os.getcwd()
    root = ROOT_IMAGES.format(CWD)
    print(image_path)
    img = Image.open(root + '/' + image_path).convert('RGB')
    im = preprocess(img)

    label = torch.tensor(label, dtype=torch.long, device=device)
    t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap = by_class_map(model, im, label,
                                                                                     operation=operation,
                                                                                     use_mask=use_mask)

    return t1, blended_img, heatmap_cv, blended_img_mask, im, score, heatmap


def by_class_map(model, image, label, operation='ours', use_mask=False):
    weight_ratio = []
    model.eval()
    model.zero_grad()
    preds = model(image.unsqueeze(0).to(device), hook=True)
    _, predicted = torch.max(preds.data, 1)

    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()
    preds.to(device)
    one_hot.to(device)
    gradients = model.get_activations_gradient()
    heatmap = grad2heatmaps(model, image.unsqueeze(0).to(device), gradients, activations=None, operation=operation,
                            score=score, do_nrm_rsz=True,
                            weight_ratio=weight_ratio)

    t = tensor2cv(image)
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, blended_img, heatmap_cv, blended_img_mask, t, score, heatmap


def image_show(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def calc_score_original(input):
    global label
    preds_original_image = model(input.to(device), hook=False).detach()
    one_hot = torch.zeros(preds_original_image.shape).to(device)
    one_hot[:, label] = 1

    score_original_image = torch.sum(one_hot * preds_original_image, dim=1).detach()
    return score_original_image


def calc_score_masked(masked_image):
    global label
    preds_masked_image = model(masked_image.unsqueeze(0).to(device), hook=False).detach()
    one_hot = torch.zeros(preds_masked_image.shape).to(device)
    one_hot[:, label] = 1
    score_masked_image = torch.sum(one_hot * preds_masked_image, dim=1).detach()
    return score_masked_image


def calc_img_score(img):
    global label
    preds_masked_image = model(img.unsqueeze(0).to(device), hook=False).detach()
    one_hot = torch.zeros(preds_masked_image.shape).to(device)
    one_hot[:, label] = 1
    score = torch.sum(one_hot * preds_masked_image, dim=1).detach()
    return score


def calc_blended_image_score(heatmap):
    global label
    img_cv = tensor2cv(input.squeeze())
    heatmap_cv = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    masked_image = np.uint8((np.repeat(heatmap_cv.reshape(224, 224, 1), 3, axis=2) * img_cv))
    img = preprocess(Image.fromarray(masked_image))
    preds_masked_image = model(img.unsqueeze(0).to(device), hook=False).detach()
    one_hot = torch.zeros(preds_masked_image.shape).to(device)
    one_hot[:, label] = 1
    score = torch.sum(one_hot * preds_masked_image, dim=1).detach()
    return score


def create_set_from_txt():
    images_by_label = {}
    with open(f'imgs/pics.txt') as f:
        lines = f.readlines()
        for line in lines:
            file_name, label = line.split()
            label = int(label)
            if label not in images_by_label:
                images_by_label[label] = [file_name]
            else:
                images_by_label[label].append(file_name)
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        label = k
        image_paths = v
        for j, image_path in enumerate(image_paths):
            set_input.append({IMAGE_PATH: image_path, LABEL: label})
    df = pd.DataFrame(set_input)

    return df


def create_set_from_txt_prod():
    images_by_label = {}
    with open(DATA_VAL_TXT) as f:
        lines = f.readlines()
        for line in lines:
            file_name, label = line.split()
            label = int(label)
            if label not in images_by_label:
                images_by_label[label] = [file_name]
            else:
                images_by_label[label].append(file_name)
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        label = k
        image_paths = v
        for j, image_path in enumerate(image_paths):
            set_input.append({IMAGE_PATH: image_path, LABEL: label})
    df = pd.DataFrame(set_input)

    return df


def handle_image_saving(blended_im, blended_img_mask, label, operation, save_image=False, save_mask=False):
    im_to_save = blended_im
    if save_mask:
        im_to_save = blended_img_mask

    if save_image:
        title = f'method: {operation}, label: {int(label)}'
        img_dict.append({"image": im_to_save, "title": title})


def write_imgs_iterate(img_name):
    num_rows = 3
    num_col = 4
    f = plt.figure(figsize=(30, 20))
    plt.subplot(num_rows, num_col, 1)
    plt.imshow(t)
    plt.title('ground truth')
    plt.axis('off')

    i = 2
    for item in img_dict:
        plt.subplot(num_rows, num_col, i)
        plt.imshow(item["image"])
        plt.title(item["title"])
        plt.axis('off')
        i += 1

    if img_name is not None:
        plt.savefig(img_name)

    plt.clf()
    plt.close('all')


def handle_sic():
    try:
        inp_img = tensor2cv(input.cpu().squeeze())
        metric_hm = heatmap
        # plt.imshow(input.cpu().squeeze().permute(1, 2, 0))
        # plt.show()

        sic = pic_metrics.calculate_sic(model, metric_hm, inp_img, label)
        print(sic.auc)
        method_sic = f'SIC_{operation}'
        current_image_results[method_sic] = sic.auc
    except Exception as e:
        print('Ignored SIC', e)
        method_sic = f'SIC_{operation}'
        current_image_results[method_sic] = 'None'


def handle_aic():
    try:
        inp_img = tensor2cv(input.cpu().squeeze())
        metric_hm = heatmap
        # print(inp_img.shape)
        aic = pic_metrics.calculate_aic(model, metric_hm, inp_img, label)
        print(aic.auc)
        method_aic = f'AIC_{operation}'
        current_image_results[method_aic] = aic.auc
    except Exception as e:
        print('Ignored AIC', e)
        method_aic = f'AIC_{operation}'
        current_image_results[method_aic] = 'None'


class ReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input, inplace=False)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def sorted_position(array):
    a = np.argsort(array)
    a[a.copy()] = np.arange(len(a))
    return a


def metric_to_number_higher_or_lower(metric):
    '''
    :param metric: a string representing the metric
    :return:
    0 - means higher is better
    1 - means lower is better
    '''
    if metric == 'PIC' or metric == 'NEG' or metric == 'INS' or metric == 'AIC' or metric == 'SIC':
        return 0
    return 1


def restart_family_best():
    family_bests['NORMAL'] = metric_best_per_sample.copy()
    family_bests['UNIFORM'] = metric_best_per_sample.copy()
    family_bests['BLUR'] = metric_best_per_sample.copy()
    family_bests['TRAIN'] = metric_best_per_sample.copy()
    family_bests['CONST'] = metric_best_per_sample.copy()


def restart_chosen_best():
    chosen_bests['CHOSEN'] = metric_best_per_sample.copy()
    chosen_bests['UNIFORM'] = metric_best_per_sample.copy()


def create_model_with_learnable_head():
    model = torchvision.models.resnet18(pretrained=True)

    # Freeze all model parameters except the final fully connected layer
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    model.fc = torch.nn.Linear(512, 100)
    model = model.to(device)
    model.eval()
    return model


def normal_sampling(x):
    samples = []
    products = []
    for i in range(5):
        try:
            normal_distribution = torch.distributions.Normal(means[i], variances[i])
        except:
            if torch.isnan(means[i]).any():
                means[i] = torch.nan_to_num(means[i], nan=0.0)
            if torch.isnan(variances[i]).any():
                variances[i] = torch.nan_to_num(variances[i], nan=1.0)
            normal_distribution = torch.distributions.Normal(means[i], variances[i])

        sample = normal_distribution.sample((1,)).squeeze().to(device)
        samples.append(sample)
        products.append(torch.dot(sample, x.squeeze()))
    sample_products = np.array(products)
    max_index = np.argmax(sample_products)
    return max_index, samples[max_index]


def get_best_value(current_image_results, metric_name, higher_lower, method):
    best_value = 0
    if higher_lower == 0:
        for k, v in current_image_results.items():
            if k.__contains__(metric_name) and k.__contains__(method):
                if v > best_value:
                    best_value = v
    else:
        best_value = 1000
        for k, v in current_image_results.items():
            if k.__contains__(metric_name) and k.__contains__(method):
                if v < best_value:
                    best_value = v
    return best_value


def save_results_to_csv_step(step):
    global df
    df = pd.DataFrame(results)
    df.loc['total'] = df.mean()
    df.loc['fields'] = df.keys()
    df.to_csv(f'./csvs/{model_name}-isbymax{BY_MAX_CLASS}-{step}.csv')


def write_heatmap(model_name, image_path, operation, heatmap_cv):
    CWD = os.getcwd()
    np.save("{0}/imgs/heatmaps/{1}_{2}_{3}".format(CWD, image_path[:-5], operation, model_name), heatmap_cv)


def write_mask(model_name, image_path, operation, masked_image):
    CWD = os.getcwd()
    np.save("{0}/imgs/masks/{1}_{2}_{3}".format(CWD, image_path[:-5], operation, model_name), masked_image)


ITERATION = BEE
models = ['densnet', 'convnext', 'resnet101', 'vit-base', 'vit-small']
layer_options = [12, 8]

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    image = None
    results = []
    model_name = models[2]
    FEATURE_LAYER_NUMBER = layer_options[1]

    PREV_LAYER = FEATURE_LAYER_NUMBER - 1
    interpolation_on_activations_steps_arr = [INTERPOLATION_STEPS]
    interpolation_on_images_steps_arr = [INTERPOLATION_STEPS]
    num_layers_options = [1]

    USE_MASK = True
    save_img = True
    save_heatmaps_masks = False
    operations = ['fullgrad', 'ablation-cam', 'lift-cam', 'layercam', 'ig', 'blurig', 'guidedig',
                  'gradcam', 'gradcampp', 'x-gradcam']
    operations = [BEE + '_' + TYPES[0], BEE + '_' + TYPES[1], BEE + '_' + TYPES[2], BEE + '_' + TYPES[3],
                  BEE + '_' + TYPES[4]]
    operations = [BEE + '_' + TYPES[0]]
    OPERATION_NUM = 3
    torch.nn.modules.activation.ReLU.forward = ReLU.forward
    if model_name.__contains__('vgg'):
        torch.nn.modules.activation.ReLU.forward = ReLU.forward

    model = GradModel(models[2], feature_layer=FEATURE_LAYER_NUMBER)
    model.to(device)
    model.eval()
    model.zero_grad()

    df = create_set_from_txt()
    df_train = df.copy()
    print(len(df))
    df_len = len(df)

    metric_learn_params = {
        'PIC': {'model': create_model_with_learnable_head(),
                'w': torch.nn.Parameter(torch.rand((1, 100)).to(device), requires_grad=True)},
        'ADP': {'model': create_model_with_learnable_head(),
                'w': torch.nn.Parameter(torch.rand((1, 100)).to(device), requires_grad=True)},
        'AIC': {'model': create_model_with_learnable_head(),
                'w': torch.nn.Parameter(torch.rand((1, 100)).to(device), requires_grad=True)},
        'SIC': {'model': create_model_with_learnable_head(),
                'w': torch.nn.Parameter(torch.rand((1, 100)).to(device), requires_grad=True)},
        'NEG': {'model': create_model_with_learnable_head(),
                'w': torch.nn.Parameter(torch.rand((1, 100)).to(device), requires_grad=True)},
        'POS': {'model': create_model_with_learnable_head(),
                'w': torch.nn.Parameter(torch.rand((1, 100)).to(device), requires_grad=True)},
        'INS': {'model': create_model_with_learnable_head(),
                'w': torch.nn.Parameter(torch.rand((1, 100)).to(device), requires_grad=True)},
        'DEL': {'model': create_model_with_learnable_head(),
                'w': torch.nn.Parameter(torch.rand((1, 100)).to(device), requires_grad=True)}
    }
    for index, row in tqdm(df.iterrows()):

        metric_distributions = {
            'PIC': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
            'ADP': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
            'AIC': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
            'SIC': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
            'NEG': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
            'POS': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
            'INS': {'trials': [0, 0, 0], 'wins': [0, 0, 0]},
            'DEL': {'trials': [0, 0, 0], 'wins': [0, 0, 0]}
        }

        restart_family_best()
        restart_chosen_best()

        for sample_repeater in range(TIMES_TO_REPEAT):
            current_image_results = {}
            image_path = row[IMAGE_PATH]
            label = row[LABEL]
            target_label = label
            input = get_images(image_path, 0)
            input_predictions = model(input.to(device), hook=False).detach()
            predicted_label = torch.max(input_predictions, 1).indices[0].item()

            if BY_MAX_CLASS:
                label = predicted_label

            res_class_saliency = run_all_operations(model, image_path=image_path,
                                                    label=label, model_name=model_name, device=device,
                                                    features_layer=FEATURE_LAYER_NUMBER,
                                                    operations=operations[OPERATION_NUM:], use_mask=USE_MASK)

            operation_index = 0
            score_original_image = 0
            img_dict = []
            for operation in operations:
                if operation.__contains__(BEE + '_' + TYPES[0]):
                    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                        get_by_class_saliency_bee(image_path=image_path,
                                                  label=label,
                                                  operations=[operation],
                                                  model_name=model_name,
                                                  layers=[FEATURE_LAYER_NUMBER],
                                                  interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
                                                  interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
                                                  device=device,
                                                  use_mask=USE_MASK, mode=TYPES[0])
                elif operation.__contains__(BEE + '_' + TYPES[1]):
                    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                        get_by_class_saliency_bee(image_path=image_path,
                                                  label=label,
                                                  operations=[operation],
                                                  model_name=model_name,
                                                  layers=[FEATURE_LAYER_NUMBER],
                                                  interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
                                                  interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
                                                  device=device,
                                                  use_mask=USE_MASK, mode=TYPES[1])
                elif operation.__contains__(BEE + '_' + TYPES[2]):
                    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                        get_by_class_saliency_bee(image_path=image_path,
                                                  label=label,
                                                  operations=[operation],
                                                  model_name=model_name,
                                                  layers=[FEATURE_LAYER_NUMBER],
                                                  interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
                                                  interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
                                                  device=device,
                                                  use_mask=USE_MASK, mode=TYPES[2])
                elif operation.__contains__(BEE + '_' + TYPES[3]):
                    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                        get_by_class_saliency_bee(image_path=image_path,
                                                  label=label,
                                                  operations=[operation],
                                                  model_name=model_name,
                                                  layers=[FEATURE_LAYER_NUMBER],
                                                  interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
                                                  interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
                                                  device=device,
                                                  use_mask=USE_MASK, mode=TYPES[3])
                elif operation.__contains__(BEE + '_' + TYPES[4]):
                    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                        get_by_class_saliency_bee_traindata(image_path=image_path,
                                                            label=label,
                                                            operations=[operation],
                                                            model_name=model_name,
                                                            layers=[FEATURE_LAYER_NUMBER],
                                                            interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
                                                            interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
                                                            device=device,
                                                            use_mask=USE_MASK, image_paths=df_train)
                elif operation == 'lift-cam':
                    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = lift_cam(
                        model,
                        image_path=image_path,
                        label=label,
                        device=device,
                        use_mask=USE_MASK)
                elif operation == 'ablation-cam':
                    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = ablation_cam_torchcam(
                        model_name,
                        image_path=image_path,
                        label=label,
                        device=device,
                        use_mask=USE_MASK)
                elif operation == 'ig':
                    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = ig(
                        model,
                        image_path=image_path,
                        label=label,
                        device=device,
                        use_mask=USE_MASK)
                elif operation == 'blurig':
                    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = blurig(
                        model,
                        image_path=image_path,
                        label=label,
                        device=device,
                        use_mask=USE_MASK)
                elif operation == 'guidedig':
                    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = guidedig(
                        model,
                        image_path=image_path,
                        label=label,
                        device=device,
                        use_mask=USE_MASK)
                elif operation == 'layercam':
                    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = layercam_torchcam(
                        model_name,
                        image_path=image_path,
                        label=label,
                        device=device,
                        use_mask=USE_MASK)
                elif operation == 'fullgrad':
                    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = fullgrad_torchcam(
                        model_name,
                        image_path=image_path,
                        label=label,
                        device=device,
                        use_mask=USE_MASK)
                else:
                    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = res_class_saliency[
                        operation_index]
                    operation_index = operation_index + 1

                handle_image_saving(blended_im, blended_img_mask, label, operation, save_image=True, save_mask=False)
                if save_heatmaps_masks:
                    write_heatmap(model_name, image_path, operation, heatmap_cv)
                    write_mask(model_name, image_path, operation, blended_img_mask)

                handle_aic()
                handle_sic()
                evaluations.run_all_evaluations(input, operation, predicted_label,
                                                target_label,
                                                save_image=save_img,
                                                heatmap=heatmap,
                                                blended_img_mask=blended_img_mask,
                                                blended_im=None, t=t,
                                                model=model,
                                                result_dict=current_image_results)

                current_image_results2 = {}

                current_image_results2[IMAGE_PATH] = image_path
                current_image_results2[LABEL] = label
                # current_image_results2[INPUT_SCORE] = score_original_image.item()

                for family in TYPES:
                    for metric_string in METRICS:
                        best_val = get_best_value(current_image_results, metric_string, HIGHER_LOWER_MAP[metric_string],
                                                  f'dix-{family.lower()}')
                        if HIGHER_LOWER_MAP[metric_string] == 0:
                            family_bests[family][metric_string] = max(family_bests[family][metric_string], best_val)
                        else:
                            family_bests[family][metric_string] = min(family_bests[family][metric_string], best_val)
                        current_image_results2[f'{family}_BEST_{metric_string}'] = family_bests[family][metric_string]

                for metric_string in METRICS:
                    input_features = metric_learn_params[metric_string]['model'](input.cuda())
                    with torch.no_grad():
                        max_index, max_sample = normal_sampling(input_features)
                    chosen_bandit = max_index
                    family_index = [0, 1, 2, 3, 4]

                    uniform_chosen = np.random.randint(4, size=1)[0]
                    family_index.remove(chosen_bandit)
                    option1 = current_image_results[f'{TYPES[family_index[0]]}_BEST_{metric_string}']
                    option2 = current_image_results[f'{TYPES[family_index[1]]}_BEST_{metric_string}']
                    option3 = current_image_results[f'{TYPES[family_index[2]]}_BEST_{metric_string}']
                    option4 = current_image_results[f'{TYPES[family_index[3]]}_BEST_{metric_string}']
                    chosen_family_result = current_image_results[f'{TYPES[chosen_bandit]}_BEST_{metric_string}']
                    current_image_results[f'{metric_string}_CHOSEN'] = chosen_family_result
                    current_image_results[f'{metric_string}_UNIFORM'] = current_image_results[
                        f'{TYPES[uniform_chosen]}_BEST_{metric_string}']

                    if HIGHER_LOWER_MAP[metric_string] == 0:
                        chosen_bests['CHOSEN'][metric_string] = max(chosen_bests['CHOSEN'][metric_string],
                                                                    chosen_family_result)
                    else:
                        chosen_bests['CHOSEN'][metric_string] = min(chosen_bests['CHOSEN'][metric_string],
                                                                    chosen_family_result)

                    if HIGHER_LOWER_MAP[metric_string] == 0:
                        chosen_bests['UNIFORM'][metric_string] = max(chosen_bests['UNIFORM'][metric_string],
                                                                     current_image_results[
                                                                         f'{TYPES[uniform_chosen]}_BEST_{metric_string}'])
                    else:
                        chosen_bests['UNIFORM'][metric_string] = min(chosen_bests['UNIFORM'][metric_string],
                                                                     current_image_results[
                                                                         f'{TYPES[uniform_chosen]}_BEST_{metric_string}'])

                    current_image_results2[f'{metric_string}_CHOSEN_{sample_repeater}'] = chosen_bests['CHOSEN'][
                        metric_string]
                    current_image_results2[f'{metric_string}_UNIFORM'] = chosen_bests['UNIFORM'][metric_string]

                    if metric_string == 'PIC':  # handles pic situation where you get only 100 or 0
                        metric_distributions[metric_string]['trials'][chosen_bandit] += 1
                        sample_rank = 0
                        if chosen_family_result > 0:
                            metric_distributions[metric_string]['wins'][chosen_bandit] += 1
                            sample_rank = 1
                        metric_score = torch.tensor(sample_rank)
                        if IS_TRAIN:
                            apply_contextual_thompson(input.cuda(), metric_learn_params[metric_string]['model'],
                                                      max_index,
                                                      metric_learn_params[metric_string]['w'], metric_score)
                        continue

                metric_history[metric_string].append(option1)
                metric_history[metric_string].append(option2)
                metric_history[metric_string].append(option3)
                metric_history[metric_string].append(option4)
                metric_history[metric_string].append(chosen_family_result)
                chosen_position = sorted_position(metric_history[metric_string])[-1]
                number_of_samples = len(metric_history[metric_string])
                chosen_rank = 1
                if number_of_samples > 1:
                    chosen_rank = chosen_position / (number_of_samples - 1)
                    print(f'chosen rank for {metric_string} is : {chosen_rank}')

                high_low = metric_to_number_higher_or_lower(metric_string)

                if high_low == 1:
                    chosen_rank = 1 - chosen_rank
                metric_distributions[metric_string]['trials'][chosen_bandit] += 1
                metric_distributions[metric_string]['wins'][chosen_bandit] += chosen_rank

                sample_rank = np.random.choice([1, -1], 1, p=[chosen_rank, 1 - chosen_rank])[0]
                metric_score = torch.tensor(sample_rank)
                if IS_TRAIN:
                    apply_contextual_thompson(input.cuda(), metric_learn_params[metric_string]['model'], max_index,
                                              metric_learn_params[metric_string]['w'], metric_score)

        if index > NUMBER_OF_IMAGES_TO_TRAIN_ON:
            IS_TRAIN = False
            TIMES_TO_REPEAT = TIMES_TO_REPEAT_TEST

        if index % 1000 == 0 and index != 0 and IS_TRAIN == False:
            save_results_to_csv_step(index)
            print(index)

        if save_img:
            string_lbl = label_map.get(int(label))
            write_imgs_iterate(f'qualitive_results/{model_name}_{string_lbl}_{image_path}')
        score_original_image = calc_score_original(input)
        torch.cuda.empty_cache()
