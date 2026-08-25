import scipy.io as sio
import os
import numpy as np
import logging
import random
from ssim_torch import ssim
import cv2
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from einops import rearrange, repeat


Test_bool = True
# Test_bool = False

def generate_masks(mask_path, batch_size, channels_numb):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, channels_numb))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

def generate_masks_ones(batch_size, channels_numb):
    mask = np.ones([256,256])
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, channels_numb))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

def shift_mask(mask3d, size=256):
    """
    input[b, 28, h, w]
    output[b, 28,h, w+2d]
    """
    b,c,w,h = mask3d.shape
    mask3d_shift = torch.zeros((b, c, size, size + (c - 1) * 2)).cuda().float()
    mask3d_shift[:,:, :, 0:size] = mask3d
    for t in range(c):
        mask3d_shift[:, t, :,:] = torch.roll(mask3d_shift[:, t, :, :], 2 * t, dims=2)
    return mask3d_shift

def sum_shift_mask(mask3d_shift):
    """
    input[b, c, h, w+2d]
    output[b, h, w+2d]
    """
    mask_3d_shift_s = torch.sum(mask3d_shift ** 2, dim=1, keepdim=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1
    return mask_3d_shift_s

def generate_shift_masks(channels_numb, mask3d, mask_path, batch_size, mask_type_setting):
    if channels_numb == 28 and mask_type_setting in ['manual', "dynamic"]:
        mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
        mask_3d_shift = mask['mask_3d_shift']
        mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
        mask_3d_shift = torch.from_numpy(mask_3d_shift)
        [nC, H, W] = mask_3d_shift.shape
        Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
        Phi_s_batch = torch.sum(Phi_batch**2,1)
        Phi_s_batch[Phi_s_batch==0] = 1
        # print(Phi_batch.shape, Phi_s_batch.shape)
    else:
        Phi_batch = shift_mask(mask3d)
        Phi_s_batch = sum_shift_mask(Phi_batch)

    return Phi_batch, Phi_s_batch

# def LoadTraining(path):
#     imgs = []
#     scene_list = os.listdir(path)
#     scene_list.sort()
#     print('training sences:', len(scene_list))
#     for i in range(len(scene_list)):
#     # for i in range(5):
#         scene_path = path + scene_list[i]
#         scene_num = int(scene_list[i].split('.')[0][5:])
#         if scene_num<=205:
#             if 'mat' not in scene_path:
#                 continue
#             img_dict = sio.loadmat(scene_path)
#             if "img_expand" in img_dict:
#                 img = img_dict['img_expand'] / 65536.
#             elif "img" in img_dict:
#                 img = img_dict['img'] / 65536.
#             img = img.astype(np.float32)
#             imgs.append(img)
#             print('Sence {} is loaded. {}'.format(i, scene_list[i]))
#     return imgs
#
# def LoadTest(path_test):
#     scene_list = os.listdir(path_test)
#     scene_list.sort()
#     test_data = np.zeros((len(scene_list), 256, 256, 28))
#     for i in range(len(scene_list)):
#         scene_path = path_test + scene_list[i]
#         img = sio.loadmat(scene_path)['img']
#         test_data[i, :, :, :] = img
#     test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
#     return test_data
# def LoadTraining(path, RGB_path):
#     imgs = []
#     RGBs = []
#     scene_list = os.listdir(path)
#     scene_list.sort()
#     print('training sences:', len(scene_list))
#     # for i in range(len(scene_list)):
#     for i in range(10):
#         scene_path = path + scene_list[i]
#         scene_RGB_path = RGB_path + scene_list[i]
#
#         scene_num = int(scene_list[i].split('.')[0][5:])
#         if scene_num<=205:
#             if 'mat' not in scene_path:
#                 continue
#             img_dict = sio.loadmat(scene_path)
#             if "img_expand" in img_dict:
#                 img = img_dict['img_expand'] / 65536.
#             elif "img" in img_dict:
#                 img = img_dict['img'] / 65536.
#             img = img.astype(np.float32)
#             imgs.append(img)
#             print('Sence {} is loaded. {}'.format(i, scene_list[i]))
#
#             RGB_dict = sio.loadmat(scene_RGB_path)
#             RGB = RGB_dict['RGB']
#             RGBs.append(RGB)
#             print('Sence {} RGB is loaded. {}'.format(i, scene_list[i]))
#     return imgs, RGBs

class LoadTraining(Dataset):
    def __init__(self, path, RGB_path, crop_size, arg=True, stride=16):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        h,w = 1024,1024  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        scene_list = os.listdir(path)
        scene_list.sort()
        print(f'traing len(scene) of CAVE dataset:{len(scene_list)}')

        list_nub = len(scene_list)

        if Test_bool == True:
            list_nub = 10
        for i in range(list_nub):
            scene_path = path + scene_list[i]
            scene_RGB_path = RGB_path + scene_list[i]
            scene_num = int(scene_list[i].split('.')[0][5:])
            if scene_num <= 205:
                if 'mat' not in scene_path:
                    continue
                img_dict = sio.loadmat(scene_path)
                if "img_expand" in img_dict:
                    hyper = img_dict['img_expand'] / 65536.
                elif "img" in img_dict:
                    hyper = img_dict['img'] / 65536.
                hyper = hyper.astype(np.float32)
                print('Sence {} is loaded. {}'.format(i, scene_list[i]))

                RGB_dict = sio.loadmat(scene_RGB_path)
                RGB = RGB_dict['RGB']
                print('Sence {} RGB is loaded. {}'.format(i, scene_list[i]))

                self.hypers.append(hyper.transpose(2, 0, 1))
                self.bgrs.append(RGB.transpose(2, 0, 1))

        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img*self.img_num

# def LoadTest(path_test, test_path_RGB):
#     scene_list = os.listdir(path_test)
#     scene_list.sort()
#     test_data = np.zeros((len(scene_list), 256, 256, 28))
#     test_RGB = np.zeros((len(scene_list), 256, 256, 3))
#     for i in range(len(scene_list)):
#     # for i in range(10):
#         scene_path = path_test + scene_list[i]
#         img = sio.loadmat(scene_path)['img']
#         test_data[i, :, :, :] = img
#
#         scene_path_RGB = test_path_RGB + scene_list[i]
#         RGB = sio.loadmat(scene_path_RGB)['RGB']
#         test_RGB[i, :, :, :] = RGB
#
#     test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
#     test_RGB = torch.from_numpy(np.transpose(test_RGB, (0, 3, 1, 2)))
#     return test_data, test_RGB
class LoadTest(Dataset):
    def __init__(self, path, RGB_path):
        # self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        # self.arg = arg
        # h,w = 1024,1024  # img shape
        # self.stride = stride
        # self.patch_per_line = (w-crop_size)//stride+1
        # self.patch_per_colum = (h-crop_size)//stride+1
        # self.patch_per_img = self.patch_per_line*self.patch_per_colum

        scene_list = os.listdir(path)
        scene_list.sort()

        list_nub = len(scene_list)
        if Test_bool == True:
            list_nub = 10
        for i in range(list_nub):
            scene_path = path + scene_list[i]
            scene_RGB_path = RGB_path + scene_list[i]
            scene_num = int(scene_list[i].split('.')[0][5:])
            if 205 < scene_num:
                if 'mat' not in scene_path:
                    continue
                img_dict = sio.loadmat(scene_path)
                if "img_expand" in img_dict:
                    hyper = img_dict['img_expand'] / 65536.
                elif "img" in img_dict:
                    hyper = img_dict['img'] / 65536.
                hyper = hyper.astype(np.float32)
                print('Sence {} is loaded.'.format(scene_list[i]))

                RGB_dict = sio.loadmat(scene_RGB_path)
                RGB = RGB_dict['RGB']
                print('Sence {} RGB is loaded.'.format(scene_list[i]))

                self.hypers.append(cv2.pyrDown(cv2.pyrDown(hyper)).transpose(2, 0, 1))
                self.bgrs.append(cv2.pyrDown(cv2.pyrDown(RGB)).transpose(2, 0, 1))

        self.img_num = len(self.hypers)
        self.length = self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        # stride = self.stride
        # crop_size = self.crop_size
        # img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        # h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[idx]
        hyper = self.hypers[idx]
        # bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        # hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        # rotTimes = random.randint(0, 3)
        # vFlip = random.randint(0, 1)
        # hFlip = random.randint(0, 1)
        # if self.arg:
        #     bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
        #     hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.img_num

class LoadVal(Dataset):
    def __init__(self, path_test, test_path_RGB):
        self.hypers = []
        self.bgrs = []
        scene_list = os.listdir(path_test)
        scene_list.sort()
        print(f'test len(hyper_test) of kaist dataset:{len(scene_list)}')
        list_nub = len(scene_list)
        if Test_bool == True:
            list_nub = 10
        for i in range(list_nub):
            scene_path = path_test + scene_list[i]
            hyper = sio.loadmat(scene_path)['img']

            scene_path_RGB = test_path_RGB + scene_list[i]
            bgr = sio.loadmat(scene_path_RGB)['RGB']
            self.hypers.append(hyper.transpose(2, 0, 1))
            self.bgrs.append(bgr.transpose(2, 0, 1))
            print(f'cave scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)

def LoadMeasurement(path_test_meas):
    img = sio.loadmat(path_test_meas)['simulation_test']
    test_data = img
    test_data = torch.from_numpy(test_data)
    return test_data

# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def shuffle_crop(train_data, batch_size, crop_size=256, argument=True):
    if argument:
        gt_batch = []
        # The first half data use the original data.
        index = np.random.choice(range(len(train_data)), batch_size//2)
        processed_data = np.zeros((batch_size//2, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size//2):
            img = train_data[index[i]]
            h, w, _ = img.shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        for i in range(processed_data.shape[0]):
            gt_batch.append(arguement_1(processed_data[i]))

        # The other half data use splicing.
        processed_data = np.zeros((4, 128, 128, 28), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                x_index = np.random.randint(0, h-crop_size//2)
                y_index = np.random.randint(0, w-crop_size//2)
                processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]
            gt_batch_2 = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()  # [4,28,128,128]
            gt_batch.append(arguement_2(gt_batch_2))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        return gt_batch

def arguement_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x

def arguement_2(generate_gt):
    c, h, w = generate_gt.shape[1],256,256
    divid_point_h = 128
    divid_point_w = 128
    output_img = torch.zeros(c,h,w).cuda()
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img

def gen_meas_torch(channels, data_batch, mask3d_batch, Y2H=True, mul_mask=False):
    nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas, nC=channels)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

def shift_back(inputs, step=2, nC = 28):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def init_mask(channels_numb, mask_type_setting, mask_path, mask_type, batch_size):
    if mask_type_setting in ['manual', "dynamic"]:
        mask3d_batch = generate_masks(mask_path, batch_size, channels_numb=channels_numb)
    elif mask_type_setting in ['mask_free']:
        mask3d_batch = generate_masks_ones(batch_size, channels_numb=channels_numb)
    if mask_type == 'Phi':
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == 'Phi_PhiPhiT':
        Phi_batch, Phi_s_batch = generate_shift_masks(channels_numb, mask3d_batch, mask_path, batch_size, mask_type_setting)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == 'Mask':
        input_mask = mask3d_batch
    elif mask_type == None:
        input_mask = None
    return mask3d_batch, input_mask


def gen_DMDC_meas(data_batch, mask3d_batch, Y2H=False, mul_mask=False, shiftback=True):
    nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    if shiftback:
        meas = shift_back(meas, nC = nC)
        return meas
    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas



def init_meas(datasets_channels_numb, gt, mask, input_setting):
    if input_setting == 'H':
        input_meas = gen_meas_torch(channels=datasets_channels_numb, data_batch=gt, mask3d_batch=mask, Y2H=True, mul_mask=False)
    elif input_setting == 'HM':
        input_meas = gen_meas_torch(channels=datasets_channels_numb, data_batch=gt, mask3d_batch=mask, Y2H=True, mul_mask=True)
    elif input_setting == 'Y':
        input_meas = gen_meas_torch(channels=datasets_channels_numb, data_batch=gt, mask3d_batch=mask, Y2H=False, mul_mask=True)
    elif input_setting == 'DMDC':  #
        input_meas = gen_DMDC_meas(gt, mask, Y2H=False, mul_mask=False, shiftback=True)
    return input_meas




def checkpoint(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

def checkpoint_mask(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_mask_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint_mask saved to {}".format(model_out_path))

class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        h,w = 482,512  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'

        with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr) of ntire2022 dataset:{len(bgr_list)}')

        list_nub = len(hyper_list)
        if Test_bool == True:
            list_nub = 10
        for i in range(list_nub):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper =np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] ==bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])  # [3,482,512]
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()
            print(f'Ntire2022 scene {i} is loaded.')
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_valid) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of ntire2022 dataset:{len(bgr_list)}')

        list_nub = len(hyper_list)
        if Test_bool == True:
            list_nub = 10
        for i in range(list_nub):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])   # [3,482,512]
            self.hypers.append(hyper[:,113:369,128:384])
            self.bgrs.append(bgr[:,113:369,128:384])
            mat.close()
            print(f'Ntire2022 scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)
from PIL import Image
# class ValidDataset(Dataset):
#     def __init__(self, data_root, bgr2rgb=True):
#         self.hypers = []
#         self.bgrs = []
#         hyper_data_path = "/home/czy/NET/spectral_mindspore/Mask_free/simulation/meas.bmp"
#         bgr_data_path = '/home/czy/NET/spectral_mindspore/Mask_free/simulation/rgb.bmp'
#         with Image.open(hyper_data_path) as img:
#             hyper = np.float32(img)[:,:,0] / 255.0
#
#         with Image.open(bgr_data_path) as img:
#             bgr = np.float32(img) / 255.0
#             bgr = np.transpose(bgr,[2,0,1])
#         self.hypers.append(hyper)
#         self.bgrs.append(bgr)
#
#
#     def __getitem__(self, idx):
#         hyper = self.hypers[idx]
#         bgr = self.bgrs[idx]
#         return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)
#
#     def __len__(self):
#         return len(self.hypers)

class TestDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        with open(f'{data_root}/split_txt/test_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_test) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr_test) of ntire2022 dataset:{len(bgr_list)}')

        list_nub = len(hyper_list)
        if Test_bool == True:
            list_nub = 10
        for i in range(list_nub):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])   # [3,482,512]
            self.hypers.append(hyper[:,113:369,128:384])
            self.bgrs.append(bgr[:,113:369,128:384])
            mat.close()
            print(f'Ntire2022 scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)



class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label.clamp(min=1e-6)
        mrae = torch.mean(error.reshape(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        # N = im_true.size()[0]
        # C = im_true.size()[1]
        # H = im_true.size()[2]
        # W = im_true.size()[3]
        # Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        # Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        # mse = nn.MSELoss(reduce=False)
        # err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        # psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        # return torch.mean(psnr)

        psnr_list= []
        for k in range(im_true.shape[0]):
            psnr_val = torch_psnr(im_true[k, :, :, :], im_fake[k, :, :, :])
            psnr_list.append(psnr_val.detach().cpu().numpy())
        psnr_mean = np.mean(np.asarray(psnr_list))
        return psnr_mean

class Loss_SSIM(nn.Module):
    def __init__(self):
        super(Loss_SSIM, self).__init__()

    def forward(self, im_true, im_fake):
        return ssim(im_true, im_fake)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_mask(mask_path):
    """ load mask
    output [h, w, 28]"""
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    # mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 1))
    mask = torch.from_numpy(mask)
    return mask

def init_rgb(rgb, size=256, dim=28):
    """
    input[b, 28, h, w]
    output[b, 28,h, w+2d]
    """
    b,c,w,h = rgb.shape
    rgb_new = torch.zeros((b, c, size, size + (dim - 1) * 2)).cuda().float()
    rgb_new[:,:, :, 0:size] = rgb
    return rgb_new

def shift_rgb(y, dim=28, step=2):
    b, c, h, w = y.shape
    rgb3d = repeat(y, 'b c h w -> b c nc h w', nc=dim)
    rgb3d = rearrange(rgb3d, 'b c nc h w -> (b c) nc h w')
    for i in range(dim):
        rgb3d[:, i, :, :] = torch.roll(rgb3d[:, i, :, :], shifts=step * i, dims=2)
    rgb3d = rearrange(rgb3d, '(b c) nc h w  -> b (c nc) h w', b=b, c=c)


    return rgb3d