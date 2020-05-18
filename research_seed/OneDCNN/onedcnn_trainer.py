"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from research_seed.OneDCNN.onedcnn import OneDCNN

from utils.inout import save_results

from settings import *

def main(hparams):
    # init module
    model = OneDCNN(hparams)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        num_sanity_val_steps=0,
        # amp_level='O1',
        # precision=16,
        # use_amp=False,
        # accumulate_grad_batches=16
    )
    trainer.fit(model)

    trainer.test()

    path = HYP_DEV_PATH
    save_results(path, trainer.model.test_results)



if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)

    parser = OneDCNN.add_model_specific_args(parser)

    hparams = parser.parse_args()

    main(hparams)
