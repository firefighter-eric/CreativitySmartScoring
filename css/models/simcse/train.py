import pytorch_lightning as pl

from css.models.simcse import args
from css.models.simcse.task import CSETask

if __name__ == '__main__':
    from css.models.simcse.dataloader import get_dataloader

    train_args = args.TrainArgs
    train_dataloader, dev_dataloader = get_dataloader(train_args)

    task = CSETask(pretrained=True)
    # task = CSETask.load_from_checkpoint('/mnt/user/zhangqifan/item_rater_dict/ridge/simcse/bert-base-chinese/lightning_logs/version_10/checkpoints/epoch=5-step=3797.ckpt')

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=3, monitor='val/loss')

    trainer = pl.Trainer(
        gpus=[0],
        num_nodes=1,
        precision=train_args.precision,
        default_root_dir=train_args.model_output_path,
        # val_check_interval=train_args.val_check_interval,
        max_epochs=train_args.epochs,
        # strategy='ddp',
        # callbacks=[checkpoint_callback],
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.validate(task, dev_dataloader)

    trainer.fit(task,
                train_dataloader,
                dev_dataloader)
