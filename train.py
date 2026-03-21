import os
import torch
import wandb
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig, ListConfig
from omegaconf.base import ContainerMetadata

# PyTorch 2.6+ requires whitelisting custom types for checkpoint loading
torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata])

from motGPT.callback import build_callbacks
from motGPT.config import parse_args, instantiate_from_config
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.utils.logger import create_logger
from motGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae

def main():
    # Configs
    cfg = parse_args(phase="train")  # parse config file

    # Logger
    logger = create_logger(cfg, phase="train")  # create logger
    logger.info(OmegaConf.to_yaml(cfg))  # print config file

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Metric Logger
    pl_loggers = []
    for loggerName in cfg.LOGGER.TYPE:
        if loggerName == 'tensorboard':
            pl_logger = instantiate_from_config(cfg.LOGGER.TENSORBOARD)
            pl_loggers.append(pl_logger)
        elif loggerName == 'wandb':
            # Initialize wandb directly instead of using PyTorch Lightning's WandbLogger
            # Only initialize on rank 0 to avoid multiple wandb runs in distributed training
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            if local_rank == 0:
                wandb_params = cfg.LOGGER.WANDB.params
                try:
                    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
                except Exception:
                    # Fallback if resolution fails due to missing interpolations
                    wandb_config = OmegaConf.to_container(cfg, resolve=False)
                wandb.init(
                    project=wandb_params.get('project', 'motiongpt3'),
                    name=cfg.NAME,
                    config=wandb_config,
                    tags=wandb_params.get('tags', []),
                    mode='offline' if wandb_params.get('offline', False) else 'online',
                    dir=cfg.FOLDER_EXP,
                )
                # Set epoch as the x-axis for all metrics
                wandb.define_metric("epoch")
                wandb.define_metric("*", step_metric="epoch")

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase='train')
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Model
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)
    
    # Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        # precision='16',
        logger=pl_loggers if pl_loggers else False,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp_find_unused_parameters_true"
        if len(cfg.DEVICE) > 1 else 'auto',
        benchmark=False,
        deterministic=False,
        accumulate_grad_batches=cfg.TRAIN.accumulate_grad_batches,
    )
    logger.info("Trainer initialized")

    # Strict load pretrianed model
    if cfg.TRAIN.PRETRAINED:
        load_pretrained(cfg, model, logger)
    # model.lm.resize_tokenizer()

    # Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)

    # Run initial validation (epoch 0) to establish baseline metrics
    if not cfg.TRAIN.RESUME:
        logger.info("Running initial validation (epoch 0) to establish baseline metrics...")
        model.initial_validation = True  # Flag for epoch 0 logging
        trainer.validate(model, datamodule=datamodule)
        model.initial_validation = False
        logger.info("Initial validation complete.")

    # Pytorch 2.0 Compile
    # if torch.__version__ >= "2.0.0":
    #     model = torch.compile(model, mode="reduce-overhead")
    # model = torch.compile(model)

    # Lightning Fitting
    if cfg.TRAIN.RESUME:
        # resume_config() sets PRETRAINED to the checkpoint file path
        trainer.fit(model,
                    datamodule=datamodule,
                    ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datamodule)

    # Training ends
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")

    # Close wandb run if it was initialized (only on rank 0)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if 'wandb' in cfg.LOGGER.TYPE and local_rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
