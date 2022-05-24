import os
import pdb
import torch
import pprint
import argparse
import pytorch_lightning as pl
from torch import autograd
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from src.core.lightning import LitModule
from src.utils.os_utils import copy_code
from src.utils.train_utils import set_seed
from src.core.config import run_grid_search_experiments

torch.multiprocessing.set_sharing_strategy('file_system')

def main(hparams):
    log_dir = hparams.LOG_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_device = torch.cuda.device_count() 
    # if num_device > 1 and hparams.TRAINING.STRATEGY is None:
    #     hparams.TRAINING.STRATEGY = 'ddp'

    set_seed(hparams.SEED_VALUE)

    logger.add(
        os.path.join(log_dir, 'train.log'),
        level='INFO',
        colorize=False,
    )

    copy_code(
        output_folder=log_dir,
        curr_folder=os.path.dirname(os.path.abspath(__file__))
    )

    logger.info(torch.cuda.get_device_properties(device))
    logger.info(f'Hyperparameters: \n {hparams}')

    logger.info('*** Building model ***')
    lit_module = LitModule(hparams=hparams)
    
    
    # Turn on PL logging and Checkpoint saving
    tb_logger = None
    ckpt_callback = False

    bar_callback = TQDMProgressBar(refresh_rate=hparams.REFRESH_RATE)
    if hparams.PL_LOGGING == True:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            verbose=True,
            save_top_k=3,
            mode='min',
            every_n_epochs=hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH,
        )
        summary_callback = ModelSummary(max_depth=5)
        # initialize tensorboard logger
        tb_logger = TensorBoardLogger(
            save_dir=log_dir,
            name='tb_logs',
        )

    # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(
        devices=num_device,
        accelerator="gpu",
        strategy=hparams.TRAINING.STRATEGY,
        callbacks=[checkpoint_callback, summary_callback, bar_callback],
        logger=tb_logger,
        default_root_dir=log_dir,
        log_every_n_steps=50,
        max_epochs=hparams.TRAINING.MAX_EPOCHS,
        check_val_every_n_epoch=hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH,
        reload_dataloaders_every_n_epochs=hparams.TRAINING.RELOAD_DATALOADERS_EVERY_N_EPOCH,
        num_sanity_val_steps=0,
        fast_dev_run=hparams.FAST_DEV_RUN,
        detect_anomaly=hparams.DETECT_ANOMALY,
    )

    if hparams.RUN_TEST:
        logger.info('*** Started testing ***')
        trainer.test(model=lit_module)
    else:
        logger.info('*** Started training ***')
        trainer.fit(lit_module, ckpt_path=hparams.TRAINING.RESUME)
        trainer.test()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--cfg_id', type=int, default=0, help='cfg id to run when multiple experiments are spawned')

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    hparams = run_grid_search_experiments(
        cfg_id=args.cfg_id,
        cfg_file=args.cfg,
        script='train.py',
    )

    main(hparams)
