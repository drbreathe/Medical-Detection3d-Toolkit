import glob
import importlib
import numpy as np
import torch.nn as nn
import os
import pandas as pd
import SimpleITK as sitk
import time
import torch
from easydict import EasyDict as edict

from detection3d.utils.file_io import load_config
from detection3d.utils.model_io import get_checkpoint_folder
from detection3d.utils.image_tools import convert_image_to_tensor, convert_tensor_to_image, \
    resample_spacing, pick_largest_connected_component, weighted_voxel_center
from detection3d.utils.normalizer import FixedNormalizer, AdaptiveNormalizer
from detection3d.dataset.landmark_dataset import read_image_list


def read_test_folder(folder_path):
    """ read single-modality input folder
    :param folder_path: image file folder path
    :return: a list of image path list, list of image case names
    """
    suffix = ['.mhd', '.nii', '.hdr', '.nii.gz', '.mha', '.image3d']
    file = []
    for suf in suffix:
        file += glob.glob(os.path.join(folder_path, '*' + suf))

    file_name_list, file_path_list = [], []
    for im_pth in sorted(file):
        _, im_name = os.path.split(im_pth)
        for suf in suffix:
            idx = im_name.find(suf)
            if idx != -1:
                im_name = im_name[:idx]
                break
        file_name_list.append(im_name)
        file_path_list.append(im_pth)

    return file_name_list, file_path_list


def load_det_model(model_folder, gpu_id=0):
    """ load segmentation model from folder
    :param model_folder:    the folder containing the segmentation model
    :param gpu_id:          the gpu device id to run the segmentation model
    :return: a dictionary containing the model and inference parameters
    """
    assert os.path.isdir(model_folder), 'Model folder does not exist: {}'.format(
        model_folder)

    # load inference config file
    latest_checkpoint_dir = get_checkpoint_folder(
        os.path.join(model_folder, 'checkpoints'), -1)
    infer_cfg = load_config(
        os.path.join(latest_checkpoint_dir, 'lmk_infer_config.py'))

    model = edict()
    model.infer_cfg = infer_cfg
    train_cfg = load_config(
        os.path.join(latest_checkpoint_dir, 'lmk_train_config.py'))
    model.train_cfg = train_cfg

    # load model state
    chk_file = os.path.join(latest_checkpoint_dir, 'params.pth')

    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(int(gpu_id))
        # load network module
        state = torch.load(chk_file)
        net_module = importlib.import_module(
            'detection3d.network.' + state['net'])
        net = net_module.Net(state['in_channels'], state['num_landmark_classes'] + 1)
        net = nn.parallel.DataParallel(net, device_ids=[0])
        net.load_state_dict(state['state_dict'])
        net.eval()
        net = net.cuda()
        del os.environ['CUDA_VISIBLE_DEVICES']

    else:
        state = torch.load(chk_file, map_location='cpu')
        net_module = importlib.import_module(
            'detection3d.network.' + state['net'])
        net = net_module.Net(state['in_channels'], state['num_landmark_classes'] + 1)
        net.load_state_dict(state['state_dict'])
        net.eval()

    model.net = net
    model.crop_size, model.crop_spacing, model.max_stride, model.interpolation = \
        state['crop_size'], state['crop_spacing'], state['max_stride'], state['interpolation']
    model.in_channels, model.num_landmark_classes = \
        state['in_channels'], state['num_landmark_classes']

    model.crop_normalizers = []
    for crop_normalizer in state['crop_normalizers']:
        if crop_normalizer['type'] == 0:
            mean, stddev, clip = crop_normalizer['mean'], crop_normalizer['stddev'], \
                                 crop_normalizer['clip']
            model.crop_normalizers.append(FixedNormalizer(mean, stddev, clip))

        elif crop_normalizer['type'] == 1:
            clip_sigma = crop_normalizer['clip_sigma']
            model.crop_normalizers.append(AdaptiveNormalizer(clip_sigma))

        else:
            raise ValueError('Unsupported normalization type.')

    return model


def detection_voi(model, iso_image, start_voxel, end_voxel, use_gpu):
    """ Segment the volume of interest
    :param model:           the loaded segmentation model.
    :param iso_image:       the image volume that has the same spacing with the model's resampling spacing.
    :param start_voxel:     the start voxel of the volume of interest (inclusive).
    :param end_voxel:       the end voxel of the volume of interest (exclusive).
    :param use_gpu:         whether to use gpu or not, bool type.
    :return:
      mean_prob_maps:        the mean probability maps of all classes
      std_maps:              the standard deviation maps of all classes
    """
    assert isinstance(iso_image, sitk.Image)

    roi_image = iso_image[start_voxel[0]:end_voxel[0],
                start_voxel[1]:end_voxel[1], start_voxel[2]:end_voxel[2]]

    if model['crop_normalizers'] is not None:
        roi_image = model.crop_normalizers[0](roi_image)

    roi_image_tensor = convert_image_to_tensor(roi_image).unsqueeze(0)
    if use_gpu:
        roi_image_tensor = roi_image_tensor.cuda()

    with torch.no_grad():
        landmarks_pred = model['net'](roi_image_tensor)

    return landmarks_pred


def detection_single_image(image, image_name, model, gpu_id, save_prob, save_folder):
    """ volumetric segmentation for single image
    :param image: the input volume
    :param image_name: the name of the image
    :param model: the detection model
    :param gpu_id: the id of the gpu
    :return: a dictionary containing the detected landmarks
    """
    assert isinstance(image, sitk.Image)

    # load landmark label dictionary
    landmark_dict = model['train_cfg'].general.target_landmark_label
    landmark_name_list, landmark_label_list = [], []
    for landmark_name in landmark_dict.keys():
        landmark_name_list.append(landmark_name)
        landmark_label_list.append(landmark_dict[landmark_name])
    landmark_label_reorder = np.argsort(landmark_label_list)

    iso_image = resample_spacing(image, model['crop_spacing'], model['max_stride'],
                                 model['interpolation'])
    assert isinstance(iso_image, sitk.Image)

    start_voxel, end_voxel = [0, 0, 0], [int(iso_image.GetSize()[idx]) for idx in range(3)]
    voi_landmarks_pred = detection_voi(model, iso_image, start_voxel, end_voxel, gpu_id > 0)
    print('{:0.2f}%'.format( 100))

    # convert to landmark masks
    landmark_mask_preds = voi_landmarks_pred.cpu()
    assert landmark_mask_preds.shape[0] == 1
    landmark_mask_preds = torch.squeeze(landmark_mask_preds)
    landmark_mask_preds = convert_tensor_to_image(landmark_mask_preds, sitk.sitkFloat32)

    detected_landmark = []
    for i in range(0, model['num_landmark_classes']):
        landmark_mask_pred = landmark_mask_preds[i + 1]  # exclude the background
        landmark_mask_pred.CopyInformation(iso_image)

        landmark_mask_prob = sitk.GetArrayFromImage(landmark_mask_pred)
        # threshold the probability map to get the binary mask
        prob_threshold = 0.5
        landmark_mask_binary = np.zeros_like(landmark_mask_prob, dtype=np.int16)
        landmark_mask_binary[landmark_mask_prob >= prob_threshold] = 1
        landmark_mask_binary[landmark_mask_prob < prob_threshold] = 0

        # pick the largest connected component
        landmark_mask_cc = sitk.GetImageFromArray(landmark_mask_binary)
        landmark_mask_cc = pick_largest_connected_component(landmark_mask_cc, [1])

        # only keep probability of the largest connected component
        landmark_mask_cc = sitk.GetArrayFromImage(landmark_mask_cc)
        masked_landmark_mask_prob = np.multiply(landmark_mask_cc.astype(np.float32), landmark_mask_prob)

        # compute the weighted mass center of the probability map
        masked_landmark_mask_prob = sitk.GetImageFromArray(masked_landmark_mask_prob)
        masked_landmark_mask_prob.CopyInformation(iso_image)
        voxel_coordinate = weighted_voxel_center(masked_landmark_mask_prob, prob_threshold, 1.0)

        landmark_name = landmark_name_list[landmark_label_reorder[i]]
        if voxel_coordinate is not None:
            world_coordinate = masked_landmark_mask_prob.TransformContinuousIndexToPhysicalPoint(voxel_coordinate)
            print("world coordinate of volume {0} landmark {1} is:[{2},{3},{4}]".format(
                image_name, i, world_coordinate[0], world_coordinate[1], world_coordinate[2]))
            detected_landmark.append(
                [landmark_name, world_coordinate[0], world_coordinate[1], world_coordinate[2]]
            )
        else:
            print("world coordinate of volume {0} landmark {1} is not detected.".format(image_name, i))
            detected_landmark.append([landmark_name, 0, 0, 0])

    detected_landmark_df = pd.DataFrame(data=detected_landmark, columns=['name', 'x', 'y', 'z'])

    return detected_landmark_df

import pandas as pd

import os
import time
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from detection3d.utils.image_tools import convert_tensor_to_image, resample_spacing, pick_largest_connected_component, weighted_voxel_center
from detection3d.utils.file_io import load_config
from detection3d.utils.model_io import get_checkpoint_folder
from detection3d.dataset.landmark_dataset import read_image_list
# from detection3d.utils import load_det_model, read_test_folder, detection_voi

def detection(input_path, model_folder, gpu_id, return_landmark_file, save_landmark_file, save_prob, output_folder, prob_threshold=0.5):
    """Volumetric landmark detection engine with confidence thresholding."""
    
    # Load model
    begin = time.time()
    model = load_det_model(model_folder, gpu_id)
    load_model_time = time.time() - begin

    # Load landmark label dictionary
    landmark_dict = model['train_cfg'].general.target_landmark_label
    landmark_name_list, landmark_label_list = zip(*landmark_dict.items())
    landmark_label_reorder = np.argsort(landmark_label_list)

    # Load test images
    if os.path.isfile(input_path):
        if input_path.endswith('.csv'):
            file_name_list, file_path_list, _, _ = read_image_list(input_path, 'test')
        else:
            file_name_list = [os.path.basename(input_path)]
            file_path_list = [input_path]
    elif os.path.isdir(input_path):
        file_name_list, file_path_list = read_test_folder(input_path)
    else:
        raise ValueError(f"Unsupported input path: {input_path}")

    if save_landmark_file or save_prob:
        os.makedirs(output_folder, exist_ok=True)

    # Collect results
    all_results = []
    for i, file_path in enumerate(file_path_list):
        file_name = file_name_list[i]
        print(f"[{i}] Processing: {file_name}")

        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue

        # Load and resample image
        start_read = time.time()
        image = sitk.ReadImage(file_path, sitk.sitkFloat32)
        iso_image = resample_spacing(image, model['crop_spacing'], model['max_stride'], model['interpolation'])
        read_image_time = time.time() - start_read

        # Forward pass
        start_voxel = [0, 0, 0]
        end_voxel = [int(iso_image.GetSize()[i]) for i in range(3)]
        start_infer = time.time()
        preds = detection_voi(model, iso_image, start_voxel, end_voxel, gpu_id is not None and gpu_id >= 0)

        infer_time = time.time() - start_infer

        preds = preds.cpu().squeeze(0)
        pred_images = convert_tensor_to_image(preds, sitk.sitkFloat32)

        # Evaluate each landmark class
        start_post = time.time()
        for j in range(model['num_landmark_classes']):
            landmark_mask_pred = pred_images[j + 1]  # skip background
            landmark_mask_pred.CopyInformation(iso_image)

            landmark_mask_prob = sitk.GetArrayFromImage(landmark_mask_pred)
            max_prob = float(np.max(landmark_mask_prob))

            # Threshold and clean prediction
            binary_mask = (landmark_mask_prob >= prob_threshold).astype(np.uint8)
            cleaned = pick_largest_connected_component(sitk.GetImageFromArray(binary_mask), [1])
            cleaned_array = sitk.GetArrayFromImage(cleaned)

            # Weight by prob
            masked_prob = cleaned_array * landmark_mask_prob
            masked_prob_image = sitk.GetImageFromArray(masked_prob)
            masked_prob_image.CopyInformation(iso_image)

            voxel_coord = weighted_voxel_center(masked_prob_image, prob_threshold, 1.0)
            landmark_name = landmark_name_list[landmark_label_reorder[j]]
            if voxel_coord is not None and max_prob >= prob_threshold:
                # Ensure Python floats, not numpy types/array
                world_coord = masked_prob_image.TransformContinuousIndexToPhysicalPoint(
                    tuple(float(v) for v in voxel_coord)
                )

                result = {
                    'file_name': file_name,
                    'landmark_name': landmark_name,
                    'detected': 1,
                    'x': world_coord[0],
                    'y': world_coord[1],
                    'z': world_coord[2],
                    'confidence': max_prob
                }
            else:
                result = {
                    'file_name': file_name,
                    'landmark_name': landmark_name,
                    'detected': 0,
                    'x': -1,
                    'y': -1,
                    'z': -1,
                    'confidence': max_prob
                }

            all_results.append(result)

            # Optionally save probability map
            if save_prob:
                sitk.WriteImage(landmark_mask_pred, os.path.join(output_folder, f"{file_name}_{landmark_name}.mha"))

        post_time = time.time() - start_post
        print(f"⏱ read: {read_image_time:.2f}s | infer: {infer_time:.2f}s | post: {post_time:.2f}s")
        
    print(all_results)
    result_df = pd.DataFrame(all_results)
    return result_df
    # if save_landmark_file:
    #     result_df.to_csv(os.path.join(output_folder, "landmark_detection_summary.csv"), index=False)

    # if return_landmark_file:
    #     return result_df
    # else:
    #     return None
