# -*- coding: utf-8 -*-
import argparse
import logging
import torch
import pytorch_lightning as pl
from src import utils
from pathlib import Path
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser()
parser.add_argument("--dir",
                    help="model directory, default: models/lfblips",
                    default=Path('models/lfblips'))
parser.add_argument("--data", help="'real' or 'fake' data, default: 'real'",
                    default='real', type=str)
args = parser.parse_args()


def main(mdir, data):
    logger = logging.getLogger(__name__)
    logger.info('Starting model testing')

    model = utils.AutoEncoder.load_from_checkpoint(f"{mdir}/params.ckpt")
    dm = utils.GlitchDataModule(data=data, width=model.width)
    trainer = pl.Trainer()
    trainer.test(datamodule=dm, model=model, ckpt_path=f"{mdir}/params.ckpt")

    logger.info('Model testing finished')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(args.dir, args.data)
