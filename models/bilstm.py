
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from .preprocessing_layer import Preprocessing

class BiLSTM(L.LightningModule):
    def __init__(self, model_type, output_features, norm_file, in_vars, extra_shape=0, hidden_size=96, n_layer=1, lr=1.e-3):
        super(BiLSTM, self).__init__()
        self.model_type = model_type
        self.extra_shape = extra_shape
        self.lr = lr
        layer_kwags = {}
        input_features = len(in_vars)
        self.prep = Preprocessing(norm_file, in_vars, mode="horizontal", pad_len=47, var_len=47)
        self.lstm = nn.LSTM(input_features, hidden_size, bidirectional=True, batch_first=True, num_layers = n_layer, **layer_kwags)

        if "SW" in model_type:
            self.conv = nn.Sequential(nn.Conv1d(hidden_size*2, output_features, kernel_size=1, **layer_kwags), nn.ReLU())
        else:
            self.conv = nn.Conv1d(hidden_size*2, output_features, kernel_size=1, **layer_kwags)

    def forward(self, full_input):
        #if isinstance(full_input, tuple):
        #    full_input, _ = full_input
        if self.extra_shape > 0:
            x, q = full_input[:,  :-self.extra_shape], full_input[:,  -self.extra_shape:]
        else:
            x = full_input
            q = torch.zeros_like(x)
        x = self.prep(x)
        x, _ = self.lstm(x)
        x = torch.permute(x, [0, 2, 1])
        output = self.conv(x)
        output = torch.permute(output, [0, 2, 1])
        
        if self.extra_shape > 0:
            output = torch.cat((output, q[:, :, None]), dim=-1)
        
        return output

    def predict(self, x):
        self.eval()
        y_hat = self(x)
        return y_hat
    
 
    def training_step(self, batch, batch_idx):
        x, y = batch[0].float(), batch[1].float()
        y_hat = self(x)[:, :, 0]
        q = y[:, :, 1]
        y = y[:, :, 0]
        mae = F.l1_loss(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        loss_3 = torch.mean(torch.std(y, dim=0)*torch.mean(torch.square(y_hat-y), dim=0))
        loss_4 = torch.mean(torch.mean(torch.square(y_hat-y), dim=0)/torch.std(y, dim=0))
        q_loss = torch.mean(torch.square(torch.mul((y_hat-y),q)))
        loss = mae + mse + loss_4 + q_loss
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=True)
    
        self.log('mae_loss', mae, sync_dist=True)
        self.log('mse_loss', mse, sync_dist=True)
        self.log('std_mse_loss', loss_3, sync_dist=True)        
        self.log('mse_std_loss', loss_4, sync_dist=True)        
        self.log('q_loss', q_loss, sync_dist=True)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[0].float(), batch[1].float()
        y_hat = self(x)[:, :, 0]
        q = y[:, :, 1]
        y = y[:, :, 0]
        mae = F.l1_loss(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        loss_3 = torch.mean(torch.std(y, dim=0)*torch.mean(torch.square(y_hat-y), dim=0))
        loss_4 = torch.mean(torch.mean(torch.square(y_hat-y), dim=0)/torch.std(y, dim=0))
        q_loss = torch.mean(torch.square(torch.mul((y_hat-y),q)))
        loss = mae + mse + loss_4 + q_loss
        self.log('val_mae_loss', mae, sync_dist=True)
        self.log('val_mse_loss', mse, sync_dist=True)
        self.log('val_std_mse_loss', loss_3, sync_dist=True)
        self.log('val_mse_std_loss', loss_4, sync_dist=True)        
        self.log('val_q_loss', q_loss, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        return loss 

    def on_after_backward(self):
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:  # log every n steps
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params.grad, self.trainer.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300, verbose=True, threshold=0.0001, threshold_mode='rel')

        return {"optimizer":optimizer, "lr_scheduler":{"scheduler": scheduler, "monitor": "val_loss"}}