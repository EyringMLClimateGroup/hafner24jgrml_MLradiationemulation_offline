
import torch
import lightning as L
from glob import glob
from utils.quick_helpers import load_from_checkpoint
# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) 

# #Custom imports
import config

torch.set_float32_matmul_precision('medium')

# Load config and data
train_args = config.setup_args_and_load_data()

train_loader = torch.utils.data.DataLoader(train_args.coarse_train, batch_size=None, num_workers=0, pin_memory=True) 
val_loader = torch.utils.data.DataLoader(train_args.coarse_val, batch_size=None, num_workers=0, pin_memory=True) 

# create
baseline_model = config.create_model(train_args.x_shape, 
                                     train_args.y_shape, 
                                     nft=train_args.norm_file, 
                                     in_vars=train_args.variables["in_vars"], 
                                     extra_shape=train_args.extra_shape, 
                                     model_type=train_args.model_type, 
                                     seed=train_args.seed)
if train_args.pretrained:
    print("loading pretrained model")
    baseline_model.load_state_dict(torch.load(train_args.pretrained_path))
elif train_args.checkpoint:
    baseline_model = config.load_from_checkpoint(train_args, train_args.extra_shape)

print(str(baseline_model))
# callbacks
early_stopping = L.pytorch.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=500, mode='min', check_on_train_epoch_end=False)
lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch')
checkpoint = L.pytorch.callbacks.ModelCheckpoint(dirpath=train_args.checkpoint_path, monitor='val_loss', mode='min', save_top_k=1, save_last=True, save_weights_only=True, filename='baseline_{epoch}')

# trainer
trainer = L.Trainer(max_epochs=-1, strategy="ddp", accelerator="gpu", devices="auto", 
                    num_sanity_val_steps=0, log_every_n_steps=10, profiler="simple",
                    callbacks=[early_stopping, lr_monitor, checkpoint],
                    fast_dev_run=train_args.dev)
trainer.fit(model=baseline_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print(baseline_model.log)

# Save model
torch.save(baseline_model.state_dict(), train_args.model_path)


# Save model without extra shape for inference
baseline_model = load_from_checkpoint(train_args, 0)

# creat a trainer if it does not exist, so lighnting can save the model
baseline_model.trainer = L.Trainer()

print(str(baseline_model))

# saving jit model for online coupling
if torch.cuda.is_available():
    device = "cuda"
    baseline_model.to(device)
    scripted_model = torch.jit.script(baseline_model)
    scripted_model.save(f"{train_args.save_folder}{train_args.model_type}_scripted_model_{device}.pt")
device = "cpu"
baseline_model.to(device)
scripted_model = torch.jit.script(baseline_model)
scripted_model.save(f"{train_args.save_folder}{train_args.model_type}_scripted_model_{device}.pt")

