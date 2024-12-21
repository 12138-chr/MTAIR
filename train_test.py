import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor

from net.my_model import MTAIR  ####my model

import lightning.pytorch as pl
import torch.nn.functional as F


import glob
import io

class IRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = MTAIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=200)

        return [optimizer], [scheduler]


def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    # # 创建文件夹用于保存降噪前的图像
    degrad_output_path = os.path.join(output_path, 'degraded_images')
    subprocess.check_output(['mkdir', '-p', degrad_output_path])
    # #

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=6) ##wgail

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            # 保存降噪前的图像到指定文件夹
            degrad_image_filename = os.path.join(degrad_output_path, clean_name[0] + '.png')
            save_image_tensor(degrad_patch, degrad_image_filename)
            #

            save_image_tensor(restored, output_path + clean_name[0] + '.png') # 保存降噪的图像到指定文件夹

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))

        return psnr.avg, ssim.avg    #wjial


def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=6)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')  # 保存降噪的图像到指定文件夹

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))

        return psnr.avg, ssim.avg    #wjial


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=3,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--denoise_path', type=str, default="test/denoise/", help='save path of test noisy images') 
    parser.add_argument('--derain_path', type=str, default="test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="test/dehaze/", help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="model.ckpt",
                        help='checkpoint save path')

    parser.add_argument('--ckpt_folder', type=str, default="mymodel_need_test",
                        help='folder containing checkpoint files')  #####动态获取该路径所有ckpt
    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_files = glob.glob(os.path.join(testopt.ckpt_folder, '*.ckpt'))    #动态获取ckpt

    number = 1
    for ckpt_name in ckpt_files:    #######

        ckpt_path = ckpt_name    ###zhe li gai
        print("正在测试---------",ckpt_name)
        print("正在测试",number)
        print("总共有",len(ckpt_files))
        number += 1

        denoise_splits = ["bsd68/"]
        # denoise_splits = ["Urban100/"]
        derain_splits = ["Rain100L/"]
        dehaze_splits = ["outdoor/"]  # wogail

        denoise_tests = []
        derain_tests = []

        testopt.denoise_path = "test/denoise/"   ##########每轮循环重置
        testopt.derain_path = "test/derain/"
        testopt.dehaze_path = "test/dehaze/"     ##########每轮循环重置




        base_path = testopt.denoise_path
        for i in denoise_splits:
            testopt.denoise_path = os.path.join(base_path, i)
            denoise_testset = DenoiseTestDataset(testopt)
            denoise_tests.append(denoise_testset)

        print("CKPT name : {}".format(ckpt_path))

        net = IRModel().load_from_checkpoint(ckpt_path).cuda()
        net.eval()

        if testopt.mode == 0:
            for testset, name in zip(denoise_tests, denoise_splits):
                print('Start {} testing Sigma=15...'.format(name))
                test_Denoise(net, testset, sigma=15)

                # print('Start {} testing Sigma=25...'.format(name))
                # test_Denoise(net, testset, sigma=25)
                #
                # print('Start {} testing Sigma=50...'.format(name))
                # test_Denoise(net, testset, sigma=50)
        elif testopt.mode == 1:
            print('Start testing rain streak removal...')
            derain_base_path = testopt.derain_path
            for name in derain_splits:
                print('Start testing {} rain streak removal...'.format(name))
                testopt.derain_path = os.path.join(derain_base_path, name)
                derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
                test_Derain_Dehaze(net, derain_set, task="derain")
        elif testopt.mode == 2:
            print('Start testing SOTS...')
            dehaze_base_path = testopt.dehaze_path
            name = dehaze_splits[0]
            testopt.dehaze_path = os.path.join(dehaze_base_path, name)
            derain_set = DerainDehazeDataset(testopt, task="dehaze", addnoise=False, sigma=15)
            test_Derain_Dehaze(net, derain_set, task="dehaze")
        elif testopt.mode == 3:
            for testset, name in zip(denoise_tests, denoise_splits):
                print('Start {} testing Sigma=15...'.format(name))

                denoise_15_psnr,denoise_15_ssim = test_Denoise(net, testset, sigma=15)

                print('Start {} testing Sigma=25...'.format(name))
                denoise_25_psnr,denoise_25_ssim = test_Denoise(net, testset, sigma=25)

                print('Start {} testing Sigma=50...'.format(name))
                denoise_50_psnr,denoise_50_ssim = test_Denoise(net, testset, sigma=50)

            derain_base_path = testopt.derain_path
            print(derain_splits)
            for name in derain_splits:
                print('Start testing {} rain streak removal...'.format(name))
                testopt.derain_path = os.path.join(derain_base_path, name)
                derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)

            derain_psnr,derain_ssim = test_Derain_Dehaze(net, derain_set, task="derain")

            print('Start testing SOTS...')
            dehaze_base_path = testopt.dehaze_path
            name = dehaze_splits[0]
            testopt.dehaze_path = os.path.join(dehaze_base_path, name)
            derain_set = DerainDehazeDataset(testopt, task="dehaze", addnoise=False, sigma=15)

            dehaze_psnr,dehaze_ssim = test_Derain_Dehaze(net, derain_set, task="dehaze")

