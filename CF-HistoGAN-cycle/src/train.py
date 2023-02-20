from __future__ import print_function
import os
import shutil

import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from parserNet import get_parser
from crcDataLoader import CRCLightningLoader
from cycle import CycleGAN

def init_seed(opt):
        '''
        Disable cudnn to maximize reproducibility
        '''
        torch.cuda.cudnn_enabled = False
        torch.manual_seed(opt.manual_seed)
        torch.cuda.manual_seed(opt.manual_seed)
        cudnn.benchmark = True


def init_experiment(opt):
    if opt.experiment is None:
        opt.experiment = '../samples'
    try:
        shutil.rmtree(opt.experiment)
    except:
        pass
    if not os.path.isdir(opt.experiment):
        os.makedirs(opt.experiment)
    if not os.path.isdir(opt.experiment + '/images'):
        os.makedirs(opt.experiment + '/images')
    if not os.path.isdir(opt.experiment + '/models'):
        os.makedirs(opt.experiment + '/models')


def main():
    #important to save things on the slurm server. Delete it if you dont use slurm
    del os.environ["SLURM_JOB_NAME"]

    options = get_parser().parse_args()

    wandBLogger = WandbLogger(project=options.project, name=options.logname)

    if not options.pretrained:
        init_experiment(options)
    init_seed(options)

    #LAMBDA
    LAMBDA = 10

    dataloader = CRCLightningLoader(options)

    model = CycleGAN(options, LAMBDA)

    #model saver
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=50,
        dirpath=options.experiment + '/models',
        filename="model-{epoch:02d}",
        save_top_k=10,
        monitor="val/total_loss"
    )

    trainer = Trainer(
        logger=wandBLogger,
        callbacks=[checkpoint_callback],
        max_epochs=options.nepochs,
        gpus=1,
        enable_checkpointing=True,
        reload_dataloaders_every_n_epochs=True,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=dataloader)

if __name__ == '__main__':
    main()