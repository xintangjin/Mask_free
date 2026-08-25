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
mask3d_batch_train, input_mask_train = init_mask(channels_numb=datasets_channels_numb, mask_type_setting=opt.mask_type, mask_path=opt.mask_path, mask_type=opt.input_mask, batch_size=opt.batch_size)
mask3d_batch_test, input_mask_test = init_mask(channels_numb=datasets_channels_numb, mask_type_setting=opt.mask_type, mask_path=opt.mask_path, mask_type=opt.input_mask, batch_size=opt.batch_size)

# dataset
print("\nloading dataset ...")
if opt.dataset_name == "CAVE":
    train_data = LoadTraining(opt.cave_path, opt.cave_path_RGB,  crop_size=opt.patch_size, stride=opt.CAVE_stride)  # [3,1024,1024]  [28,1024,1024]
    test_data = LoadTest(opt.cave_path, opt.cave_path_RGB)  # [3,256,256] [28,256,256]
    val_data = LoadVal(opt.kaist_val_path, opt.kaist_val_path_RGB)  # [3,256,256] [28,256,256]
elif opt.dataset_name == "ARAD_1k":
    train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.ARAD_1k_stride) #[3,482,512] ,[31,82,512]
    print(f"Iteration per epoch: {len(train_data)}")
    test_data = TestDataset(data_root=opt.data_root, bgr2rgb=True) # [3,256,256] [31,256,256]
    print("Test set samples: ", len(test_data))
    val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True) #[3,256,256] [31,256,256]
    print("Validation set samples: ", len(val_data))


# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)


# mask
if opt.use_dynamic_mask == True:
    mask_base = load_mask(opt.mask_path).cuda()  # [w, h]
    model_mask = mask_generator(mask_base, opt.pretrained_mask_path).cuda()
# model
if opt.method=='hdnet':
    model, FDL_loss = model_generator(datasets_channels_numb, opt.method, opt.pretrained_model_path).cuda()
else:
    model = model_generator(datasets_channels_numb, opt.method, opt.pretrained_model_path).cuda()
model.load_state_dict(torch.load('/home/czy/NET/mask-free/Mask_free/simulation/train_code/exp/CAVE/mask_free/CST_l_plus without loss2/model_epoch_197.pth'))


# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.use_dynamic_mask == True:
    optimizer_mask = torch.optim.Adam(model_mask.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    if opt.use_dynamic_mask == True:
        scheduler_mask = torch.optim.lr_scheduler.MultiStepLR(optimizer_mask, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
    if opt.use_dynamic_mask == True:
        scheduler_mask = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mask, opt.max_epoch, eta_min=1e-6)

if opt.loss_type=='MSE':
    get_loss = torch.nn.MSELoss().cuda()
elif opt.loss_type=='CharbonnierLoss':
    get_loss = losses.CharbonnierLoss().cuda()
get_loss2 = losses.CurvatureLoss(type="diff").cuda()
get_loss3 = losses.CurvatureLoss(type="vector").cuda()
get_loss4 = torch.nn.MSELoss().cuda()

criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_ssim = Loss_SSIM()
criterion_mrae.cuda()
criterion_rmse.cuda()
criterion_psnr.cuda()
criterion_ssim.cuda()
def test(val_loader, model, input_mask_test, mask3d_batch_test, epoch, logger):
    truth_list, pred_list = [], []

    mrae = AverageMeter()
    rmse = AverageMeter()
    psnr = AverageMeter()
    SSIM = AverageMeter()
    begin = time.time()
    PRED = []
    TRUTH = []

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
        begin = time.time()
        with torch.no_grad():
            if opt.method in ['cst_s', 'cst_m', 'cst_l']:
                model_out, _ = model(input_meas, input_mask_test)
            elif opt.method.find('HRFT') >= 0:
                model_out = model(input_meas, test_RGB, input_mask_test[0])
            elif opt.method.find('DMDC') >= 0:
                model_out = model(input_meas, test_RGB, input_mask_test)
            elif opt.method.find('MaskfreeDC') >= 0:
                test_RGB = init_rgb(test_RGB, dim=datasets_channels_numb)
                test_RGB_s = shift_rgb(test_RGB, dim=datasets_channels_numb)
                model_out,  rgb_out = model(input_meas, input_mask_test, test_RGB, test_RGB_s)
            elif opt.method.find ('ASMNET') >= 0:
                model_out = model(input_meas, test_RGB, input_mask_test)
            elif opt.method.find ('dauhdc') >= 0:
                model_out = model(input_meas, test_RGB, input_mask_test)
            elif opt.method.find ('LTRN') >= 0:
                model_out = model(input_meas, test_RGB, input_mask_test)
            elif opt.method.find ('LADE') >= 0:
                model_out = model(input_meas, input_mask_test, test_gt)
            else:
                # model_out = model(input_meas, input_mask_test)
                model_out = model(input_meas, input_mask_test)
            loss_mrae = criterion_mrae(model_out[:, :, :, :], test_gt[:, :, :, :])
            loss_rmse = criterion_rmse(model_out[:, :, :, :], test_gt[:, :, :, :])
            # loss_psnr = criterion_psnr(model_out[:, :, :, :], test_gt[:, :, :, :])
            loss_psnr = psnr_loss(model_out[:, :, :, :].clamp(0., 1.).data.cpu().numpy(),
                                  test_gt[:, :, :, :].clamp(0., 1.).data.cpu().numpy())
            # loss_psnr = 0.1
            loss_ssim = criterion_ssim(model_out[:, :, :, :], test_gt[:, :, :, :])
            pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
            truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
            PRED.append(pred)
            TRUTH.append(truth)


            mrae.update(loss_mrae.data)
            rmse.update(loss_rmse.data)
            # psnr.update(loss_psnr.data)
            psnr.update(loss_psnr)
            SSIM.update(loss_ssim.data)
            truth_list.append(np.transpose(test_gt.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32))
            pred_list.append(np.transpose(model_out.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32))
    end = time.time()

    PRED = np.vstack(PRED)
    TRUTH = np.vstack(TRUTH)
    name = './Test_result.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': TRUTH, 'pred': PRED})


    logger.info('===> Epoch {}: testing mrae = {:.6f}, rmse = {:.6f}, psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'
                .format(epoch, mrae.avg, rmse.avg, psnr.avg, SSIM.avg,(end - begin)))
    model.train()
    pred_list = np.vstack(pred_list)
    truth_list = np.vstack(truth_list)
    return pred_list, truth_list, mrae.avg, rmse.avg, psnr.avg, SSIM.avg, end-begin



def main(mask3d_batch_train, input_mask_train, input_mask_test,mask3d_batch_test):
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    flag = True
    psnr_max = 0
    iteration = 0
    record_mrae_loss = 1000
    begin = time.time()
    # for epoch in range(3):
    for epoch in range(1, opt.max_epoch + 1):
        if opt.use_dynamic_mask == True:
            model_mask.train()
        model.train()

        # load data
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # # train
        # for i, (gt_rgb, gt_batch) in enumerate(train_loader):
        #     gt = Variable(gt_batch).cuda().float()  # [b, channel, w, h]
        #     rgb = Variable(gt_rgb).cuda().float()  # [b, 3, w, h]
        #     rgb_ori = rgb
        #     lr = optimizer.param_groups[0]['lr']
        #     optimizer.zero_grad()
        #
        #     if opt.method.find('MaskfreeDC') >= 0:
        #         rgb = init_rgb(rgb,dim=datasets_channels_numb)
        #         rgb_s = shift_rgb(rgb,dim=datasets_channels_numb)
        #         input_meas = init_meas(datasets_channels_numb, gt, mask3d_batch_train, opt.input_setting)
        #         model_out, rgb_out = model(input_meas, input_mask_train, rgb, rgb_s)
        #         loss1 = get_loss(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         # loss1 = get_loss(model_out, gt)
        #         loss2 = get_loss2(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         loss3 = 0.15 * get_loss3(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         loss4 = get_loss4(rgb_out, rgb_ori) *4
        #         # loss = loss1
        #         loss = loss1 + loss2 + loss3 +loss4
        #         if i % 2000 == 0:
        #             print(str(loss1)+'_'+str(loss2) +'_'+str(loss3)+'_'+str(loss4))
        #             # print(str(loss1) )
        #     elif opt.method.find('HRFT') >= 0:
        #         if opt.use_dynamic_mask == True:
        #             optimizer_mask.zero_grad()
        #             mask_out = model_mask(rgb)  # [b, w, h]
        #         else:
        #             mask_out = torch.repeat_interleave(mask_base.unsqueeze(0), opt.batch_size, dim=0)
        #
        #         mask3d = torch.repeat_interleave(mask_out.unsqueeze(1), datasets_channels_numb, dim=1)  # [b, 31, w, h]
        #         Phi = shift_mask(mask3d)
        #
        #         input_meas = init_meas(datasets_channels_numb, gt, mask3d, opt.input_setting)
        #         model_out = model(input_meas, rgb, Phi)
        #         loss = get_loss(model_out, gt)
        #
        #     elif opt.method.find('DMDC') >= 0:
        #         if opt.use_dynamic_mask == True:
        #             optimizer_mask.zero_grad()
        #             mask_out = model_mask(rgb)  # [b, w, h]
        #         else:
        #             mask_out = torch.repeat_interleave(mask_base.unsqueeze(0), opt.batch_size, dim=0)
        #
        #         mask3d = torch.repeat_interleave(mask_out.unsqueeze(1), datasets_channels_numb, dim=1)  # [b, 31, w, h]
        #         Phi = shift_mask(mask3d)
        #         Phi_s = sum_shift_mask(Phi)
        #
        #         if opt.input_setting == 'DMDC':
        #             Phi_s = shift_back(Phi_s, nC=datasets_channels_numb)
        #             input_mask_train = (mask3d, Phi_s)
        #         else:
        #             input_mask_train = (Phi, Phi_s)
        #
        #         input_meas = init_meas(datasets_channels_numb, gt, mask3d, opt.input_setting)
        #         model_out = model(input_meas, rgb, input_mask_train)
        #         output_meas = init_meas(datasets_channels_numb, model_out, mask3d_batch_train, opt.input_setting)
        #
        #         if opt.loss:
        #                 loss = get_loss(model_out, gt) + 0.01 * get_loss(output_meas, input_meas)
        #         else:
        #             loss = get_loss(model_out, gt)
        #
        #     elif opt.method in ['cst_s', 'cst_m', 'cst_l']:
        #         input_meas = init_meas(datasets_channels_numb, gt, mask3d_batch_train, opt.input_setting)
        #         model_out, diff_pred = model(input_meas, input_mask_train)
        #         loss = get_loss(model_out, gt)
        #         diff_gt = torch.mean(torch.abs(model_out.detach() - gt),dim=1, keepdim=True)  # [b,1,h,w]
        #         loss_sparsity = F.mse_loss(diff_gt, diff_pred)
        #         loss = loss + 2 * loss_sparsity
        #     elif opt.method in ['ASMNET']:
        #         input_meas = init_meas(datasets_channels_numb, gt, mask3d_batch_train, opt.input_setting)
        #         model_out = model(input_meas, rgb, input_mask_train)
        #         # loss = get_loss(model_out, gt)
        #         # loss1 = get_loss(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         loss1 = get_loss(model_out, gt)
        #         loss2 = get_loss2(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         loss3 = 0.15 * get_loss3(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         # loss = loss1
        #         loss = loss1 + loss2 + loss3
        #         if i % 2000 == 0:
        #             # print(str(loss1)+'_'+str(loss2) +'_'+str(loss3))
        #             print(str(loss1) )
        #     elif opt.method in ['LTRN']:
        #         input_meas = init_meas(datasets_channels_numb, gt, mask3d_batch_train, opt.input_setting)
        #         model_out = model(input_meas, rgb, input_mask_train)
        #         # loss = get_loss(model_out, gt)
        #         loss1 = get_loss(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         # loss1 = get_loss(model_out, gt)
        #         loss2 = get_loss2(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         loss3 = 0.15 * get_loss3(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         # # loss = loss1
        #         loss = loss1 + loss2 + loss3
        #         if i % 2000 == 0:
        #             # print(str(loss1)+'_'+str(loss2) +'_'+str(loss3))
        #             print(str(loss) )
        #     elif opt.method.find('LADE') >= 0:
        #         input_meas = init_meas(datasets_channels_numb, gt, mask3d_batch_train, opt.input_setting)
        #         model_out = model(input_meas, input_mask_train, gt)
        #         # loss = get_loss(model_out, gt)
        #         loss1 = get_loss(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         # loss1 = get_loss(model_out, gt)
        #         loss2 = get_loss2(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         loss3 = 0.15 * get_loss3(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         # loss = loss1
        #         loss = loss1 + loss2 + loss3
        #         if i % 2000 == 0:
        #             # print(str(loss1)+'_'+str(loss2) +'_'+str(loss3))
        #             print(str(loss1) )
        #     else:
        #         input_meas = init_meas(datasets_channels_numb, gt, mask3d_batch_train, opt.input_setting)
        #         model_out = model(input_meas, input_mask_train)
        #         # loss = get_loss(model_out, gt)
        #         loss1 = get_loss(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         # loss1 = get_loss(model_out, gt)
        #         loss2 = get_loss2(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         loss3 = 0.15 * get_loss3(model_out.reshape(-1,datasets_channels_numb), gt.reshape(-1,datasets_channels_numb))
        #         loss = loss1
        #         # loss = loss1 + loss2 + loss3
        #         if i % 2000 == 0:
        #             print(str(loss1)+'_'+str(loss2) +'_'+str(loss3))
        #             # print(str(loss1) )
        #         if opt.method == 'hdnet':
        #             fdl_loss = FDL_loss(model_out, gt)
        #             loss = loss + 0.7 * fdl_loss
        #
        #     loss.backward()
        #     losses.update(loss.data)
        #     optimizer.step()
        #     if opt.use_dynamic_mask == True:
        #         optimizer_mask.step()
        #     iteration = iteration+1
        #     if iteration % 500 == 0:
        #         end = time.time()-begin
        #         begin = time.time()
        #         logger.info(
        #             '===> iteration = {:d} | Epoch {}：  learning rate : {:.9f} , train_losses.avg={:.9f} , time: {:.4f}'
        #             .format(iteration, epoch, lr, losses.avg, end))
        #     # if iteration % 10 == 0:
        #     #     mrae_mean, rmse_mean, psnr_mean, ssim_mean, recon_time = test(val_loader, model, input_mask_test)
        #     #     print(psnr_mean)
        #test
        pred, truth, mrae_mean, rmse_mean, psnr_mean, ssim_mean, recon_time= test(test_loader, model, input_mask_test, mask3d_batch_test,epoch,logger)

        scheduler.step()
        if opt.use_dynamic_mask == True:
            scheduler_mask.step()
        logger.info(
            '===> Epoch {}： testing psnr = {:.2f}, ssim = {:.4f}, mrae: {:.9f}, rmse: {:.9f} , reconstruct_time: {:.4f}'
            .format(epoch, psnr_mean, ssim_mean, mrae_mean, rmse_mean, recon_time))


        # val
        pred, truth, mrae_mean, rmse_mean, psnr_mean, ssim_mean, recon_time = test(val_loader, model, input_mask_test,
                                                                                   mask3d_batch_test, epoch, logger)
        logger.info('--------------------------------------------')
        logger.info('     Val:       ')
        logger.info(
            '===> Epoch {}： val psnr = {:.2f}, ssim = {:.4f}, mrae: {:.9f}, rmse: {:.9f} , reconstruct_time: {:.4f}'
                .format(epoch, psnr_mean, ssim_mean, mrae_mean, rmse_mean, recon_time))
        logger.info('--------------------------------------------')


        if psnr_mean > psnr_max:
            if flag:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}_Truth'.format(epoch, psnr_max, ssim_mean) + '.mat'
                scio.savemat(name, {'truth': truth})
                flag = False
            name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
            scio.savemat(name, {'pred': pred })

            psnr_max = psnr_mean
            checkpoint(model, epoch, model_path, logger)
            if opt.use_dynamic_mask == True:
                checkpoint_mask(model_mask, epoch, model_path, logger)
            # # val
            # pred, truth, mrae_mean, rmse_mean, psnr_mean, ssim_mean, recon_time = test(val_loader, model, input_mask_test,
            #                                                               mask3d_batch_test, epoch,logger)
            # logger.info('--------------------------------------------')
            # logger.info('     Val:       ')
            # logger.info(
            #     '===> Epoch {}： testing psnr = {:.2f}, ssim = {:.4f}, mrae: {:.9f}, rmse: {:.9f} , reconstruct_time: {:.4f}'
            #         .format(epoch, psnr_mean, ssim_mean, mrae_mean, rmse_mean, recon_time))
            # logger.info('--------------------------------------------')

        elif mrae_mean < record_mrae_loss:
            name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
            scio.savemat(name, {'pred': pred })

            record_mrae_loss = mrae_mean
            checkpoint(model, epoch, model_path, logger)
            if opt.use_dynamic_mask == True:
                checkpoint_mask(model_mask, epoch, model_path, logger)




if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main(mask3d_batch_train, input_mask_train,input_mask_test, mask3d_batch_test)


