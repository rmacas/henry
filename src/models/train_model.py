# -*- coding: utf-8 -*-
import argparse
import logging
import torch
import pytorch_lightning as pl
from src import utils
from pathlib import Path
torch.set_float32_matmul_precision('medium')


parser = argparse.ArgumentParser()
parser.add_argument("--outdir",
                    help="output model dir, default: models/lfblips",
                    default=Path('models/lfblips'))
parser.add_argument("--data", help="'real' or 'fake' data, default: 'real'",
                    default='real', type=str)
parser.add_argument("--width",
                    help="first/last layer size for encoder/decoder",
                    default=64, type=int)
parser.add_argument("--dim", help="latent dimension size for the autoencoder",
                    default=5, type=int)
parser.add_argument("--lr", help="learning rate", default=1e-3, type=float)
parser.add_argument("--epochs", help="no of epochs to train the network",
                    default=100, type=int)
args = parser.parse_args()


def main(outdir, data, width, dim, lr, epochs):
    logger = logging.getLogger(__name__)
    logger.info('Starting model training')

    outdir = Path(outdir)
    utils.chdir(outdir, logger, create=True)

    ae = utils.AutoEncoder(width, dim, lr)
    dm = utils.GlitchDataModule(data=data, width=width)

    trainer = pl.Trainer(max_epochs=epochs, accelerator="auto",
                         default_root_dir=outdir)
    trainer.fit(model=ae, datamodule=dm)
    trainer.save_checkpoint(f"{outdir}/params.ckpt")

    logger.info('Model training finished')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(args.outdir, args.data, args.width, args.dim, args.lr, args.epochs)
