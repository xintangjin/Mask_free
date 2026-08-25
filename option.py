import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
# parser.add_argument('--template', default='birnat', help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--dataset_name', type=str, default='CAVE', help='CAVE or ARAD_1k')
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')
parser.add_argument("--patch_size", type=int, default=256, help="patch size")

# Data specifications CAVE
parser.add_argument("--CAVE_stride", type=int, default=192, help="CAVE stride")
# Data specifications ARAD_1k
parser.add_argument("--ARAD_1k_stride", type=int, default=112, help="ARAD_1k stride")

#mask
parser.add_argument('--mask_type', type=str, default='mask_free', help='manual, mask_free or dynamic')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi',
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask


# Saving specifications
parser.add_argument('--outf', type=str, default='test', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='cst_l_plus', help='method name, ASMNET, LADE_3stg, LTRN')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument('--use_dynamic_mask', type=bool, default=False, help='use dynamic mask')
parser.add_argument('--pretrained_mask_path', type=str, default=None, help='pretrained model directory')
parser.add_argument('--loss', type=bool, default=False, help='use measurement loss')

# Training specifications
parser.add_argument('--batch_size', type=int, default=8, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=150, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=4000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.00040)
parser.add_argument('--loss_type', type=str, default="CharbonnierLoss", help='MSE or CharbonnierLoss')


opt = parser.parse_args()
template.set_template(opt)

# dataset
opt.cave_path = f"{opt.data_root}/cave_1024_28/"
opt.cave_path_RGB = f"{opt.data_root}/cave_1024_28_RGB/"
# opt.kaist_test_path = f"{opt.data_root}/TSA_simu_data/test_Truth/"
# opt.cave_test_path_RGB = f"{opt.data_root}//cave_1024_28/"
opt.kaist_val_path = f"{opt.data_root}/TSA_simu_data/Truth/"
opt.kaist_val_path_RGB = f"{opt.data_root}/TSA_simu_data/Truth_RGB/"

opt.ARAD_1k_path = f"{opt.data_root}/Train_Spec/"
opt.ARAD_1k_path_RGB = f"{opt.data_root}/Train_RGB/"
opt.ARAD_1k_train_split_txt = f"{opt.data_root}/split_txt/train_list.txt"
opt.ARAD_1k_test_split_txt = f"{opt.data_root}/split_txt/valid_list.txt"

opt.mask_path = f"{opt.data_root}/TSA_simu_data/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False