from architecture import *
from utils import *
import torch
import scipy.io as scio
import time
import os
import numpy as np
from torch.autograd import Variable
import datetime
from option import opt
import losses
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from torch.utils.data import DataLoader


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# datasets channels
if opt.dataset_name == "CAVE":
    datasets_channels_numb = 28
elif opt.dataset_name == "ARAD_1k":
    datasets_channels_numb = 31

mask_base = torch.ones([256,256]).cuda()

# init mask

mask3d_batch_test, input_mask_test = init_mask(channels_numb=datasets_channels_numb, mask_type_setting=opt.mask_type, mask_path=opt.mask_path, mask_type=opt.input_mask, batch_size=opt.batch_size)

# dataset
print("\nloading dataset ...")
if opt.dataset_name == "CAVE":
    # test_data = LoadTest(opt.cave_path, opt.cave_path_RGB)  # [3,256,256] [28,256,256]
    val_data = LoadVal(opt.kaist_val_path, opt.kaist_val_path_RGB)  # [3,256,256] [28,256,256]
elif opt.dataset_name == "ARAD_1k":
    # test_data = TestDataset(data_root=opt.data_root, bgr2rgb=True) # [3,256,256] [31,256,256]
    val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True) #[3,256,256] [31,256,256]
    print("Validation set samples: ", len(val_data))


# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result_test/'
if not os.path.exists(result_path):
    os.makedirs(result_path)



# mask
if opt.use_dynamic_mask == True:
    mask_base = load_mask(opt.mask_path).cuda()  # [w, h]
    model_mask = mask_generator(mask_base, opt.pretrained_mask_path).cuda()
# model
if opt.method=='hdnet':
    model, FDL_loss = model_generator(datasets_channels_numb, opt.method, opt.pretrained_model_path).cuda()
else:
    model = model_generator(datasets_channels_numb, opt.method, opt.pretrained_model_path).cuda()


# optimizing

criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_ssim = Loss_SSIM()
criterion_mrae.cuda()
criterion_rmse.cuda()
criterion_psnr.cuda()
criterion_ssim.cuda()
def test(val_loader, model, input_mask_test, mask3d_batch_test):
    truth_list, pred_list = [], []
    rgb_list, rgb_pre_list = [], []
    rgb_out = None
    mrae = AverageMeter()
    rmse = AverageMeter()
    psnr = AverageMeter()
    SSIM = AverageMeter()
    begin = time.time()

    for i, (test_RGB, test_gt) in enumerate(val_loader):
        test_RGB = test_RGB.cuda().float()
        test_gt = test_gt.cuda().float()

        if opt.method.find('DMDC') >= 0:

            if opt.use_dynamic_mask == True:
                with torch.no_grad():
                    mask_out = model_mask(test_RGB)
            else:
                mask_out = torch.repeat_interleave(mask_base.unsqueeze(0), test_RGB.shape[0], dim=0)

            mask3d = torch.repeat_interleave(mask_out.unsqueeze(1), 31, dim=1)
            Phi = shift_mask(mask3d)
            Phi_s = sum_shift_mask(Phi)
            if opt.input_setting == 'DMDC':
                Phi_s = shift_back(Phi_s, nC=datasets_channels_numb)
                input_mask_test = (mask3d, Phi_s)
            else:
                input_mask_test = (Phi, Phi_s)
        else:
            if opt.use_dynamic_mask == True:
                model_mask.eval()
                with torch.no_grad():
                    mask_out = model_mask(test_RGB)
                    mask3d_batch_test = torch.repeat_interleave(mask_out.unsqueeze(1), datasets_channels_numb, dim=1)
                    Phi = shift_mask(mask3d_batch_test)
                    Phi_s = sum_shift_mask(Phi)
                    if opt.input_setting == 'DMDC':
                        Phi_s = shift_back(Phi_s)
                        input_mask_test = (mask3d_batch_test, Phi_s)

        model.eval()
        if test_gt.shape[0] != mask3d_batch_test.shape[0]:
            mask3d_batch_test = mask3d_batch_test[:test_gt.shape[0], :, :, :]
            if input_mask_test != None:
                try:
                    mask3d_batch1, input_mask1 = input_mask_test
                    mask3d_batch1 = mask3d_batch1[:test_gt.shape[0], :, :, :]
                    input_mask1 = input_mask1[:test_gt.shape[0], :, :]
                    input_mask_test = mask3d_batch1, input_mask1
                except:
                    mask3d_batch1= input_mask_test
                    mask3d_batch1 = mask3d_batch1[:test_gt.shape[0], :, :, :]
                    input_mask_test = mask3d_batch1

        input_meas = init_meas(datasets_channels_numb, test_gt, mask3d_batch_test, opt.input_setting)
        # input_meas = test_gt
        begin = time.time()
        with torch.no_grad():
            if opt.method in ['cst_s', 'cst_m', 'cst_l']:
                model_out, _ = model(input_meas, input_mask_test)
            elif opt.method.find('DMDC') >= 0:
                model_out = model(input_meas, test_RGB, input_mask_test)
            elif opt.method.find('MaskfreeDC') >= 0:
                test_RGB_cpu = test_RGB.detach().cpu().numpy()
                test_RGB = init_rgb(test_RGB, dim=datasets_channels_numb)
                test_RGB_s = shift_rgb(test_RGB, dim=datasets_channels_numb)
                model_out,  rgb_out = model(input_meas, input_mask_test, test_RGB, test_RGB_s)
            else:
                model_out = model(input_meas, input_mask_test)
            # loss_mrae = criterion_mrae(model_out[:, :, :, :], test_gt[:, :, :, :])
            # loss_rmse = criterion_rmse(model_out[:, :, :, :], test_gt[:, :, :, :])
            # # loss_psnr = criterion_psnr(model_out[:, :, :, :], test_gt[:, :, :, :])
            # loss_psnr = psnr_loss(model_out[:, :, :, :].clamp(0., 1.).data.cpu().numpy(),
            #                       test_gt[:, :, :, :].clamp(0., 1.).data.cpu().numpy())
            # # loss_psnr = 0.1
            # loss_ssim = criterion_ssim(model_out[:, :, :, :], test_gt[:, :, :, :])

            # mrae.update(loss_mrae.data)
            # rmse.update(loss_rmse.data)
            # # psnr.update(loss_psnr.data)
            # psnr.update(loss_psnr)
            # SSIM.update(loss_ssim.data)
            truth_list.append(test_gt.detach().cpu().numpy().astype(np.float32))
            # pred_list.append(np.transpose(model_out.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32))
            pred = model_out.detach().cpu().numpy()

            pred_list.append(pred.astype(np.float32))

            if rgb_out is not None:
                rgb_cpu = rgb_out.detach().cpu().numpy()
                rgb_list.append(rgb_cpu.astype(np.float32))
                rgb_pre_list.append(test_RGB_cpu.astype(np.float32))
    end = time.time()
    rgb_list = np.concatenate(rgb_list, axis=0)
    rgb_pre_list = np.concatenate(rgb_pre_list, axis=0)
    scio.savemat(result_path + 'Test_RGB.mat', {'truth': rgb_pre_list, 'pred': rgb_list})
    # print('===> testing mrae = {:.6f}, rmse = {:.6f}, psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
    #             .format(mrae.avg, rmse.avg, psnr.avg, SSIM.avg,(end - begin)))

    # pred_list = np.vstack(pred_list)
    # truth_list = np.vstack(truth_list)
    # return pred_list, truth_list, mrae.avg, rmse.avg, psnr.avg, SSIM.avg, end-begin
    return pred_list, truth_list



def main(input_mask_test,mask3d_batch_test):
    # test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # test
    # pred, truth, mrae_mean, rmse_mean, psnr_mean, ssim_mean, recon_time = test(val_loader, model, input_mask_test, mask3d_batch_test)
    pred, truth = test(val_loader, model, input_mask_test, mask3d_batch_test)

    name = result_path + 'Test_result.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': truth, 'pred': pred})



if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main(input_mask_test, mask3d_batch_test)


