import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
import dask
from dask.cache import Cache
import time
# Custom imports
import config
from utils.quick_helpers import load_from_checkpoint

manager = mp.Manager()
shared_cache = manager.dict()

def main():
    dask.config.set(scheduler="threads", num_workers=4)
    torch.set_float32_matmul_precision('medium')
    
    # Load config and data
    train_args = config.setup_args_and_load_data(cache=shared_cache)
    print(train_args.folder)
    print("*"*50)
    for k in train_args._get_kwargs():
        if "norm" in k[0] or "grid" in k[0]:
            continue
        print("* ", k)    
    print("*"*50)
    
    model = config.create_model(train_args, extra_shape=train_args.extra_shape)

    if train_args.pretrained:
        print("loading pretrained model")
        model.load_state_dict(torch.load(train_args.pretrained_path))
    elif train_args.checkpoint:
        print("loading checkpoint model")
        model = load_from_checkpoint(train_args, train_args.extra_shape)

    print(str(model))
    # callbacks
    early_stopping = L.pytorch.callbacks.EarlyStopping(monitor='val_mse_mae_loss', 
                                                       min_delta=0.00001, 
                                                       patience=150, 
                                                       mode='min', 
                                                       check_on_train_epoch_end=False)
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch')
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(dirpath=train_args.checkpoint_path, 
                                                     monitor='val_mse_mae_loss',
                                                     mode='min', 
                                                     save_top_k=1, 
                                                     save_last=True, 
                                                     filename='model_{epoch}')
    logger = TensorBoardLogger("lightning_logs", 
                               name = train_args.folder, 
                               version = train_args.model_type)
    
    #model = torch.compile(model)
    # trainer
    strategy = DDPStrategy(find_unused_parameters=True) if "FLUX_HR" in train_args.model_type else DDPStrategy()
    trainer = L.Trainer(max_epochs=train_args.train_epochs, 
                        strategy=strategy, 
                        accelerator="gpu", 
                        devices="auto", 
                        num_sanity_val_steps=0, 
                        log_every_n_steps=10, 
                        profiler="simple",
                        callbacks=[early_stopping, lr_monitor, checkpoint],
                        fast_dev_run=train_args.dev, 
                        logger=logger)
    
    train_loader = DataLoader(train_args.coarse_train,
                            batch_size=None, # xbatcher is doing the batching
                            shuffle=False,
                            num_workers=47,
                            prefetch_factor=11,
                            persistent_workers=True,
                            pin_memory=True,
                            multiprocessing_context=mp.get_context('spawn')
                            )	
    
    val_loader = DataLoader(train_args.coarse_val, 
                            batch_size=None,
                            num_workers=16,
                            prefetch_factor=8,
                            persistent_workers=True,
                            pin_memory=True,
                            multiprocessing_context=mp.get_context('spawn')
                            )

    if train_args.checkpoint:
        ckpt_path = train_args.checkpoint_path + "last.ckpt"
        trainer.fit(model=model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader, 
                    ckpt_path=ckpt_path)
    else: 
        trainer.fit(model=model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader)
    print(model.log)

    # Save model
    torch.save(model.state_dict(), train_args.model_path)
    
    # Save model without extra shape for inference
    model = config.load_model(train_args, 0)
    # create a trainer if it does not exist, so lightning can save the model
    model.trainer = L.Trainer()
    print(str(model))

    # saving jit model for online coupling
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
        scripted_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
        scripted_model.save(f"{train_args.save_folder}{train_args.model_type}_scripted_model_{device}.pt")
    device = "cpu"
    model.to(device)
    scripted_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
    scripted_model.save(f"{train_args.save_folder}{train_args.model_type}_scripted_model_{device}.pt")

if __name__ == '__main__':
    main()

