
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from .preprocessing_layer import Preprocessing

   
class Flux(L.LightningModule):
    def __init__(self, model_type, in_nodes, nft, in_vars, n_out_nodes=2,  extra_shape = 0, n_nodes=256, n_layers=2, lr=1.e-4):
        super(Flux, self).__init__()
        self.extra_shape = extra_shape
        self.model_type = model_type
        self.lr = lr
        self.prep = Preprocessing(nft, in_vars, mode="vertical")
        layer_list = [ nn.Linear(in_nodes, n_nodes), nn.Tanh()]
        for i in range(n_layers-1):
            layer_list.append(nn.Linear(n_nodes, n_nodes))
            layer_list.append(nn.Tanh())

        layer_list.append(nn.Linear(n_nodes, n_out_nodes))
        layer_list.append(nn.ReLU())
        self.layers = nn.Sequential(*layer_list)
        self.relu = nn.ReLU()

    def forward(self, full_input):
        if self.extra_shape>0:
            x, q = full_input[:, :-self.extra_shape], full_input[:, -self.extra_shape:]
        else:
            x = full_input
            q = torch.zeros_like(x)
        x = self.prep(x)
        output = self.layers(x)
        if self.model_type == "SW_FLUX":
            output = torch.minimum(torch.ones_like(output), output)
        else:
            output = output
        if self.extra_shape>0:
            output = torch.cat((output, q), dim=-1)
        return output
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.extra_shape>0:
            if "SW" in self.model_type:
                rsds_pred = y_hat[:, 0]
                rvds_dir = y_hat[:, 4]
                rvds_dif = y_hat[:, 5]
                rnds_dir = y_hat[:, 6]
                rnds_dif = y_hat[:, 7]
                rsus = y[:, -2]
                alb = y[:, -1]

                sw_loss_dirdif = torch.mean(torch.abs(rvds_dir + rvds_dif + rnds_dir + rnds_dif - rsds_pred))
                sw_loss_alb = torch.mean(torch.abs(rsds_pred * alb - rsus))
                self.log('sw_loss_alb', sw_loss_alb, sync_dist=True)
                self.log('sw_loss_dirdif', sw_loss_dirdif, sync_dist=True)
                sw_loss =  0 #sw_loss_alb     # sw_loss_dirdif +
            y_hat = y_hat[:,:-self.extra_shape]
            y = y[:,:-self.extra_shape]
        else:
            sw_loss = 0
        
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        mbe = torch.abs(torch.mean(y_hat - y)) # mean bias error
        loss = mse + mae + mbe + sw_loss
        self.log('mse_loss', mse)
        self.log('mae_loss', mae)
        self.log('mbe', mbe)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.extra_shape>0:
            if "SW" in self.model_type:
                rsds_pred = y_hat[:, 0]
                rvds_dir = y_hat[:, 4]
                rvds_dif = y_hat[:, 5]
                rnds_dir = y_hat[:, 6]
                rnds_dif = y_hat[:, 7]
                rsus = y[:, -2]
                alb = y[:, -1]

                sw_loss_dirdif = torch.mean(torch.abs(rvds_dir + rvds_dif + rnds_dir + rnds_dif - rsds_pred))
                sw_loss_alb = torch.mean(torch.abs(rsds_pred * alb - rsus))
                self.log('val_sw_loss_alb', sw_loss_alb, sync_dist=True)
                self.log('val_sw_loss_dirdif', sw_loss_dirdif, sync_dist=True)
                sw_loss = 0 #sw_loss_alb #sw_loss_dirdif + 
            y_hat = y_hat[:,:-self.extra_shape]
            y = y[:,:-self.extra_shape]
        else:
            sw_loss = 0
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        mbe = torch.abs(torch.mean(y_hat - y)) # mean bias error
        loss = mse + mae + mbe + sw_loss 
        self.log('val_mse_loss', mse, sync_dist=True)
        self.log('val_mae_loss', mae, sync_dist=True)
        self.log('val_mbe', mbe, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        return loss 
    
    def predict(self, x):
        self.eval()
        y_hat = self(x)
        return y_hat
    
    def on_after_backward(self):
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:  # log every n steps
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params.grad, self.trainer.global_step)
        self.log("lr", self.scheduler.optimizer.param_groups[0]['lr'])

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300, threshold=0.0001, threshold_mode='rel')

        return {"optimizer":optimizer, "lr_scheduler":{"scheduler": self.scheduler, "monitor": "val_loss"}}

