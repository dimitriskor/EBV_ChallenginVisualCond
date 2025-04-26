import argparse
from sys import exec_prefix
import torch
import numpy as np
from os.path import join
import os
import cv2
import random
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from utils.util import ensure_dir, flow2bgr_np
from model import model as model_arch
from data_loader.data_loaders import InferenceDataLoader
from model.model import ColorNet
from utils.util import CropParameters, get_height_width, torch2cv2, \
                       append_timestamp, setup_output_folder
from utils.timers import CudaTimer
from utils.henri_compatible import make_henri_compatible

from parse_config import ConfigParser

model_info = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def legacy_compatibility(args, checkpoint):
    assert not (args.e2vid and args.firenet_legacy)
    if args.e2vid:
        args.legacy_norm = True
        final_activation = 'sigmoid'
    elif args.firenet_legacy:
        args.legacy_norm = True
        final_activation = ''
    else:
        return args, checkpoint
    # Make compatible with Henri saved models
    if not isinstance(checkpoint.get('config', None), ConfigParser) or args.e2vid or args.firenet_legacy:
        checkpoint = make_henri_compatible(checkpoint, final_activation)
    if args.firenet_legacy:
        checkpoint['config']['arch']['type'] = 'FireNet_legacy'
    return args, checkpoint


def load_model(checkpoint):
    config = checkpoint['config']
    config['arch']['args']['unet_kwargs']['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']*2
    state_dict = checkpoint['state_dict']

    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']*2
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']*2
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    logger.info(model)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    # model.load_state_dict(state_dict)

    model = model.to(device)
    model.train()
    if args.color:
        model = ColorNet(model)
    # for param in model.parameters():
    #     param.requires_grad = False

    return model


def main(args, model, state_dict):
    dataset_kwargs = {'transforms': {},
                    'max_length': None,
                    'sensor_resolution': None,
                    'num_bins': 10,
                    'filter_hot_events': args.filter_hot_events,
                    'voxel_method': {'method': args.voxel_method,
                                    'k': args.k,
                                    't': args.t,
                                    'sliding_window_w': args.sliding_window_w,
                                    'sliding_window_t': args.sliding_window_t}
                    }
    if args.update:
        print("Updated style model")
        dataset_kwargs['combined_voxel_channels'] = False

    if args.legacy_norm:
        print('Using legacy voxel normalization')
        dataset_kwargs['transforms'] = {'LegacyNorm': {}}
    
    data_loader = InferenceDataLoader(args.events_file_path, dataset_kwargs=dataset_kwargs, ltype=args.loader_type)
    try:
        model.load_state_dict(state_dict)
        print('Loaded model')
    except:
        torch.save(model.state_dict(), '../My_model/model_filter_small_short_correct_deriv.pt')
        pass
    height, width = get_height_width(data_loader)
    ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    model_info['input_shape'] = height, width
    crop = CropParameters(width, height, model.num_encoders)

    ts_fname = setup_output_folder(args.output_folder)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0, amsgrad=True)
    train_images = f'mapped_images/{args.folder_name}/'
    for i in range(len(data_loader)):

        if i >= len(data_loader)//10:
            continue
        idx = random.randint(0, len(data_loader)-30)
        model.reset_states()
        with torch.no_grad():
            for j in range(15):
                current = data_loader.dataset.__getitem__(idx+j)
                voxel = current['events'].float().unsqueeze(0).to(device)
                model(voxel)
        outputs = []
        next_items = []
        gray_images = []
        for j in range(15, 30):
            current = data_loader.dataset.__getitem__(idx+j)
            voxel = current['events'].float().unsqueeze(0).to(device)
            outputs.append(model(voxel)['image'])
            next_items.append(data_loader.dataset.__getitem__(idx+j))
            try:
                train_frame = cv2.imread(train_images+f'{idx+j+1:06d}.png')
                gray_image = cv2.cvtColor(train_frame, cv2.COLOR_BGR2GRAY)
                gray_image = torch.tensor(gray_image/255).to(device)
                gray_images.append(gray_image)
            except:
                continue

        gray_images = torch.stack(gray_images).unsqueeze(1)
        outputs = torch.cat(outputs)
        res = 1 - ssim_loss(torch.diff(torch.diff(outputs, dim=-1), dim=-2), torch.diff(torch.diff(gray_images, dim=-1), dim=-2))
        # image = output['image'] + gray_image
        image = crop.crop(outputs[-1].detach().cpu())
        image = torch2cv2(image)
        fname = 'frame_{:05d}.png'.format(idx)
        cv2.imwrite(join(args.output_folder, fname), image)
        res.backward()
        optimizer.step()
        optimizer.zero_grad()
        append_timestamp(ts_fname, fname, next_items[-1]['timestamp'].item())
        print(f'{i}/{len(data_loader)//10}', res.item(), end = '\r')
    torch.save(model.state_dict(), '../My_model/model_filter_small_short_correct_deriv.pt')
    print('Saved model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--events_file_path', required=True, type=str,
                        help='path to events (HDF5)')
    parser.add_argument('--output_folder', default="/tmp/output", type=str,
                        help='where to save outputs to')
    parser.add_argument('--device', default='0', type=str,
                        help='indices of GPUs to enable')
    parser.add_argument('--is_flow', action='store_true',
                        help='If true, save output to flow npy file')
    parser.add_argument('--update', action='store_true',
                        help='Set this if using updated models')
    parser.add_argument('--color', action='store_true', default=False,
                      help='Perform color reconstruction')
    parser.add_argument('--voxel_method', default='between_frames', type=str,
                        help='which method should be used to form the voxels',
                        choices=['between_frames', 'k_events', 't_seconds'])
    parser.add_argument('--k', type=int,
                        help='new voxels are formed every k events (required if voxel_method is k_events)')
    parser.add_argument('--sliding_window_w', type=int,
                        help='sliding_window size (required if voxel_method is k_events)')
    parser.add_argument('--t', type=float,
                        help='new voxels are formed every t seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--sliding_window_t', type=float,
                        help='sliding_window size in seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--loader_type', default='H5', type=str,
                        help='Which data format to load (HDF5 recommended)')
    parser.add_argument('--filter_hot_events', action='store_true',
                        help='If true, auto-detect and remove hot pixels')
    parser.add_argument('--legacy_norm', action='store_true', default=False,
                        help='Normalize nonzero entries in voxel to have mean=0, std=1 according to Rebecq20PAMI and Scheerlinck20WACV.'
                        'If --e2vid or --firenet_legacy are set, --legacy_norm will be set to True (default False).')
    parser.add_argument('--e2vid', action='store_true', default=False,
                        help='set required parameters to run original e2vid as described in Rebecq20PAMI')
    parser.add_argument('--firenet_legacy', action='store_true', default=False,
                        help='set required parameters to run legacy firenet as described in Scheerlinck20WACV (not for retrained models using updated code)')
    parser.add_argument('--folder_name')

    args = parser.parse_args()
    
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    print('Loading checkpoint: {} ...'.format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path)
    args, checkpoint = legacy_compatibility(args, checkpoint)
    model = load_model(checkpoint)
    try:
        state_dict = (torch.load('../My_model/model_filter_small_short_correct_deriv.pt', weights_only=True))
    except:
        state_dict = None
    main(args, model, state_dict)



'''
import argparse
from sys import exec_prefix
import torch
import numpy as np
from os.path import join
import os
import cv2
import random
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from utils.util import ensure_dir, flow2bgr_np
from model import model as model_arch
from data_loader.data_loaders import InferenceDataLoader
from model.model import ColorNet
from utils.util import CropParameters, get_height_width, torch2cv2, \
                       append_timestamp, setup_output_folder
from utils.timers import CudaTimer
from utils.henri_compatible import make_henri_compatible

from parse_config import ConfigParser

model_info = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def legacy_compatibility(args, checkpoint):
    assert not (args.e2vid and args.firenet_legacy)
    if args.e2vid:
        args.legacy_norm = True
        final_activation = 'sigmoid'
    elif args.firenet_legacy:
        args.legacy_norm = True
        final_activation = ''
    else:
        return args, checkpoint
    # Make compatible with Henri saved models
    if not isinstance(checkpoint.get('config', None), ConfigParser) or args.e2vid or args.firenet_legacy:
        checkpoint = make_henri_compatible(checkpoint, final_activation)
    if args.firenet_legacy:
        checkpoint['config']['arch']['type'] = 'FireNet_legacy'
    return args, checkpoint


def load_model(checkpoint):
    config = checkpoint['config']
    config['arch']['args']['unet_kwargs']['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']*2
    state_dict = checkpoint['state_dict']

    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']*2
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']*2
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    logger.info(model)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    # model.load_state_dict(state_dict)

    model = model.to(device)
    model.train()
    if args.color:
        model = ColorNet(model)
    # for param in model.parameters():
    #     param.requires_grad = False

    return model


def main(args, model, state_dict):
    dataset_kwargs = {'transforms': {},
                    'max_length': None,
                    'sensor_resolution': None,
                    'num_bins': 10,
                    'filter_hot_events': args.filter_hot_events,
                    'voxel_method': {'method': args.voxel_method,
                                    'k': args.k,
                                    't': args.t,
                                    'sliding_window_w': args.sliding_window_w,
                                    'sliding_window_t': args.sliding_window_t}
                    }
    if args.update:
        print("Updated style model")
        dataset_kwargs['combined_voxel_channels'] = False

    if args.legacy_norm:
        print('Using legacy voxel normalization')
        dataset_kwargs['transforms'] = {'LegacyNorm': {}}
    
    data_loader = InferenceDataLoader(args.events_file_path, dataset_kwargs=dataset_kwargs, ltype=args.loader_type)
    try:
        model.load_state_dict(state_dict)
        print('Loaded model')
    except:
        torch.save(model.state_dict(), '../My_model/model_filter_short.pt')
        pass
    height, width = get_height_width(data_loader)
    ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    model_info['input_shape'] = height, width
    crop = CropParameters(width, height, model.num_encoders)

    ts_fname = setup_output_folder(args.output_folder)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0, amsgrad=True)
    train_images = f'mapped_images/{args.folder_name}/'
    for i in range(len(data_loader)):

        if i >= len(data_loader)//6:
            continue
        idx_1 = random.randint(0, len(data_loader)-40)
        idx_2 = random.randint(0, len(data_loader)-40)
        model.reset_states()
        with torch.no_grad():
            for j in range(30):
                current_1 = data_loader.dataset.__getitem__(idx_1+j)
                current_2 = data_loader.dataset.__getitem__(idx_2+j)
                voxel_1 = current_1['events'].float().to(device)
                voxel_2 = current_2['events'].float().to(device)
                voxel = torch.cat([voxel_1, voxel_2], dim=0)
                output = model(voxel)
        for j in range(30, 39):
            current_1 = data_loader.dataset.__getitem__(idx_1+j)
            current_2 = data_loader.dataset.__getitem__(idx_2+j)
            voxel_1 = current_1['events'].float().to(device)
            voxel_2 = current_2['events'].float().to(device)
            voxel = torch.cat([voxel_1, voxel_2], dim=0)
            output = model(voxel)

        next_item_1 = data_loader.dataset.__getitem__(idx_1+39)
        next_item_2 = data_loader.dataset.__getitem__(idx_2+39)
        next_item = torch.cat([next_item_1, next_item_2], dim=0)
        voxel = next_item['events'].float().to(device)
        try:
            train_frame_1 = cv2.imread(train_images+f'{idx_1+40:06d}.png')
            train_frame_2 = cv2.imread(train_images+f'{idx_2+40:06d}.png')
            gray_image_1 = cv2.cvtColor(train_frame_1, cv2.COLOR_BGR2GRAY)
            gray_image_2 = cv2.cvtColor(train_frame_2, cv2.COLOR_BGR2GRAY)
            gray_image = torch.stack([gray_image_1, gray_image_2], dim=0)

        except:
            continue
        # gray_image = cv2.resize(gray_image, (640, 480), interpolation = cv2.INTER_LINEAR)

        gray_image = torch.tensor(gray_image/255).to(device)
        if not args.color:
            voxel = crop.pad(voxel)
        with CudaTimer('Inference'):
            output = model(voxel)
        image = output['image'].detach().cpu()
        # res = 0.1*torch.sum(torch.square(output['image'] - gray_image))
        # res = res + 2*torch.sum(torch.square(torch.diff(torch.diff(output['image'][0][0], dim=1), dim=0) - torch.diff(torch.diff(gray_image, dim=1), dim=0)))
        # res = 1 - ssim_loss(torch.diff(torch.diff(output['image'][0][0], dim=1), dim=0).unsqueeze(0).unsqueeze(0), torch.diff(torch.diff(gray_image, dim=1), dim=0).unsqueeze(0).unsqueeze(0))
        res = 1 - ssim_loss(output['image'], gray_image.unsqueeze(1))
        output['image']
        # image = output['image'] + gray_image
        image = crop.crop(image.detach().cpu())
        image = torch2cv2(image)
        fname = 'frame_{:05d}.png'.format(idx)
        cv2.imwrite(join(args.output_folder, fname), image)
        res.backward()
        optimizer.step()
        optimizer.zero_grad()
        append_timestamp(ts_fname, fname, next_item['timestamp'].item())
        print(f'{i}/{len(data_loader)//6}', res.item(), end = '\r')
    torch.save(model.state_dict(), '../My_model/model_filter_short.pt')
    print('Saved model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--events_file_path', required=True, type=str,
                        help='path to events (HDF5)')
    parser.add_argument('--output_folder', default="/tmp/output", type=str,
                        help='where to save outputs to')
    parser.add_argument('--device', default='0', type=str,
                        help='indices of GPUs to enable')
    parser.add_argument('--is_flow', action='store_true',
                        help='If true, save output to flow npy file')
    parser.add_argument('--update', action='store_true',
                        help='Set this if using updated models')
    parser.add_argument('--color', action='store_true', default=False,
                      help='Perform color reconstruction')
    parser.add_argument('--voxel_method', default='between_frames', type=str,
                        help='which method should be used to form the voxels',
                        choices=['between_frames', 'k_events', 't_seconds'])
    parser.add_argument('--k', type=int,
                        help='new voxels are formed every k events (required if voxel_method is k_events)')
    parser.add_argument('--sliding_window_w', type=int,
                        help='sliding_window size (required if voxel_method is k_events)')
    parser.add_argument('--t', type=float,
                        help='new voxels are formed every t seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--sliding_window_t', type=float,
                        help='sliding_window size in seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--loader_type', default='H5', type=str,
                        help='Which data format to load (HDF5 recommended)')
    parser.add_argument('--filter_hot_events', action='store_true',
                        help='If true, auto-detect and remove hot pixels')
    parser.add_argument('--legacy_norm', action='store_true', default=False,
                        help='Normalize nonzero entries in voxel to have mean=0, std=1 according to Rebecq20PAMI and Scheerlinck20WACV.'
                        'If --e2vid or --firenet_legacy are set, --legacy_norm will be set to True (default False).')
    parser.add_argument('--e2vid', action='store_true', default=False,
                        help='set required parameters to run original e2vid as described in Rebecq20PAMI')
    parser.add_argument('--firenet_legacy', action='store_true', default=False,
                        help='set required parameters to run legacy firenet as described in Scheerlinck20WACV (not for retrained models using updated code)')
    parser.add_argument('--folder_name')

    args = parser.parse_args()
    
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    print('Loading checkpoint: {} ...'.format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path)
    args, checkpoint = legacy_compatibility(args, checkpoint)
    model = load_model(checkpoint)
    try:
        state_dict = (torch.load('../My_model/model_filter_short.pt', weights_only=True))
    except:
        state_dict = None
    main(args, model, state_dict)

'''