import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter



from models.vision_mamba import MambaUnet
# from models.ConvKANeXt import ConvKANeXt as KANUSeg
from models.UNet import UNet
from models.ATTUNet import AttU_Net
from models.DenseUnet import Dense_Unet as DenseUnet
from models.SwinUnet import SwinUnet, SwinUnet_config
from models.TransUnet import get_transNet as TransUNet
from models.ConvUNext import ConvUNeXt



import argparse
from engine import *
import os
import sys
from utils import *
from pprint import pprint

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str,
                    default='ConvUNeXt', help='choose the model. UNet, DenseUnet, KANUSeg, AttU_Net, ConvUNeXt, mamba_UNet, SwinUnet, TransUNet.')
parser.add_argument('--datasets', type=str,
                    default='PH2', help='choose the dataset. PH2, isic16, BUSI, GLAS, CVC-ClinicDB, Kvasir-SEG, 2018DSB.')
args = parser.parse_args()


# Save parsed arguments into a separate file for easy access
with open('parsed_args.txt', 'w') as f:
    f.write(f"network={args.network}\n")
    f.write(f"datasets={args.datasets}\n")

from config import get_config
from configs.config_setting import setting_config


def main(config):


    print(torch.cuda.current_device())




    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    



    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)




    print('#----------Prepareing Model----------#')
    
    if config.network == 'mamba_UNet':
        model_cfg = config.model_config
        model = MambaUnet(model_cfg,128,1)
        model.load_from(model_cfg)
    elif config.network == 'SwinUnet':
        model_cfg = SwinUnet_config()
        model = SwinUnet(model_cfg,img_size=224,num_classes=1)
    elif config.network == 'KANUSeg':
        model = KANUSeg(3,1)
    elif config.network == 'ConvUNeXt':
        model = ConvUNeXt(3,1)
    elif config.network == 'TransUNet':
        model = TransUNet(1)
    elif config.network == 'UNet':
        model = UNet(3,1)       
    elif config.network == 'DenseUnet':
        model = DenseUnet(3,1)  
    elif config.network == 'AttU_Net':
        model = AttU_Net(3,1).to('cuda')  
    else: raise Exception('network in not right!')
    model = model.cuda()


    cal_params_flops(model, 224, logger)





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    min_loss = 999
    min_miou = 0
    start_epoch = 1
    min_epoch = 1





    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        # checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        # min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
        min_loss = checkpoint.get('min_loss', 999)
        min_epoch = checkpoint.get('min_epoch', 1)
        loss = checkpoint.get('loss', 1.0)

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        loss,miou = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )

        if miou > min_miou:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_miou = miou
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

        # torch.save({
        #     'epoch': epoch,
        #     'min_epoch': min_epoch,
        #     'min_loss': min_loss,
        #     'loss': loss,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict(),
        #     }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-miou{min_miou:.4f}.pth')
        )      


if __name__ == '__main__':
    config = setting_config
    print("\n=== Config Settings ===")
    pprint({k: v for k, v in vars(config).items() if not k.startswith('__')})
    main(config)