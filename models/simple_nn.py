
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from .preprocessing_layer import Preprocessing
import numpy as np

class SimpleNN(L.LightningModule):
    def __init__(self, model_type, input_features, output_features, norm_file, in_vars, extra_shape=0, hidden_size=128, n_layer=1, lr=1.e-3, weight_decay=0):
        super(SimpleNN, self).__init__()
        self.model_type = model_type
        self.extra_shape = extra_shape
        self.lr = lr
        self.weight_decay = weight_decay
        self.lstm_output = torch.Tensor()
        self.prep = Preprocessing(norm_file, in_vars, mode="vertical", pad_len=47, var_len=47)
        
        if "SW" in model_type:
            self.linear = nn.Sequential(
                nn.Linear(input_features, hidden_size),
                nn.Tanh(), 
                nn.Linear(hidden_size, hidden_size), 
                nn.Tanh())
            self.output = nn.Sequential(
                nn.Linear(hidden_size, output_features), 
                nn.ReLU())
        else:
            self.linear = nn.Sequential(
                nn.Linear(input_features, hidden_size),
                nn.Tanh(), 
                nn.Linear(hidden_size, hidden_size), 
                nn.Tanh())
            self.output = nn.Sequential(
                nn.Linear(hidden_size, output_features))

    def forward(self, full_input):
        if self.extra_shape > 0:
            x = full_input[:,  :-self.extra_shape]
        else:
            x = full_input
        x = self.prep(x)
        x = self.linear(x)
        self.lstm_output = x
        output = self.output(x)
        return output.squeeze()

    def predict(self, x):
        self.eval()
        y_hat = self(x)
        return y_hat
 
    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch, v="val_")
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        return loss 
    
    def loss(self, batch, v=""):
        x, y = batch[0].float(), batch[1].float()
        y_hat = self(x)
        e = x[:, -self.extra_shape: ]
        l = y.shape[-1]
        q = e[:, :l]
        qconv = e[:, l:2*l]
        q_r = y_hat/86400/qconv # hr_hat in K/s because qconv converts K/s to W/m^2
        fnet_hr = torch.sum(q_r, dim=-1)
        mae = F.l1_loss(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        loss_3 = torch.mean(torch.std(y, dim=0)*torch.mean(torch.square(y_hat-y), dim=0))
        loss_4 = torch.mean(torch.mean(torch.square(y_hat-y), dim=0)/torch.std(y, dim=0))
        q_loss = torch.mean(torch.square(torch.mul((y_hat-y),q)))
        if "SW" in self.model_type:
            rsdt = x[:,l*3]  # toa
            alb = x[:,l*3+1] # albedo
            rsds = e[:,-2]
            rsut = e[:,-1]
            #rsus = rsds*alb
            fnet_flux = ((1-rsut)-(1-alb)*rsds)*rsdt # denormalized net-flux
        elif "LW" in self.model_type:
            rlds = e[:,-2]
            rlut = e[:,-1]
            ts_rad = x[:,3*l]
            sig = 5.670374419e-8
            em = 0.996
            rlus = em   #*sig*ts_rad**4
            fnet_flux = (-rlut-(rlds-rlus))*sig*ts_rad**4 # denormalized net-flux
        energy =  torch.mean(torch.abs(fnet_hr - fnet_flux))
        start_step = 300 #30000
        n_steps = 10 #5000
        step = self.current_epoch # self.trainer.global_step
        if step > start_step:
            loss = mse + mae + energy * np.minimum( 1.e-8*10**((step-start_step)/n_steps), 1.e-1)
            # increase the weight of the energy loss as the model converges
            # a factor of 10 every 100 epochs
        else:
           loss = mse + mae #+ q_loss
        #loss = mse #mse + mae + q_loss + loss_4 #mae + mse + loss_4 + q_loss
        self.log(f'{v}mae_loss', mae, sync_dist=True)
        self.log(f'{v}mse_mae_loss', mse + mae, sync_dist=True)
        self.log(f'{v}mse_loss', mse, sync_dist=True)
        self.log(f'{v}std_mse_loss', loss_3, sync_dist=True)
        self.log(f'{v}mse_std_loss', loss_4, sync_dist=True)        
        self.log(f'{v}energy_loss', energy, sync_dist=True)
        self.log(f'{v}q_loss', q_loss, sync_dist=True)
        return loss

    def on_after_backward(self):
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:  # log every n steps
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params.grad, self.trainer.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, threshold=0.0001, threshold_mode='rel')
        return {"optimizer":optimizer, "lr_scheduler":{"scheduler": scheduler, "monitor": "val_mse_loss"}}
    
class SimpleNN_with_Flux(L.LightningModule):
    def __init__(self, hr_model, model_type, output_features, extra_shape=0, hidden_size=96, lr=1.e-3, weight_decay=0.0, shap=False):
        super(SimpleNN_with_Flux, self).__init__()
        self.model_type = model_type
        self.extra_shape = extra_shape
        self.output_len = output_features
        self.lr = lr
        self.weight_decay = weight_decay
        self.hr_model = hr_model
        self.shap = shap
        l = 47
        if "SW" in model_type:
            self.n_e = 4 # for the partial albedos
        else:
            self.n_e = 0  
        self.linear = nn.Sequential(nn.Linear(hidden_size, 47), nn.Tanh())
        if "SW" in self.model_type:
            out_act = nn.Hardtanh(min_val=0., max_val=1.)
        else:
            out_act = nn.Hardtanh(min_val=0., max_val=2.)
        n_hidden = 32
        self.out = nn.Sequential(nn.Linear(l+self.n_e, n_hidden), nn.Tanh(), nn.Linear(n_hidden, output_features), out_act)
        
    def forward(self, full_input):
        #full_input = torch.tensor(full_input)
        if self.extra_shape > 0:
            x, q = full_input[:,  :-self.extra_shape], full_input[:,  -self.extra_shape:]
        else:
            x = full_input
            q = torch.zeros_like(x)
        if "SW" in self.model_type:
            n_e = self.n_e
            e = x[:,-n_e:]
            x = x[:,:-n_e]
        else:
            e = torch.zeros_like(x)
        
        if self.shap:
            hr = self.hr_model(x)
            lstm_output = self.hr_model.lstm_output
        else:
            with torch.no_grad():
                hr = self.hr_model(x)
                lstm_output = self.hr_model.lstm_output

        feat = self.linear(lstm_output).squeeze()
        if "SW" in self.model_type:
            flux_feat = torch.cat((feat, e), dim=-1)  
        else:
            flux_feat = feat
        output = self.out(flux_feat)
        output = torch.cat((hr.squeeze(), output), dim=-1) # return only hr and flux
        
        return output

    def predict(self, x):
        self.eval()
        y_hat = self(x)
        return y_hat 
    
 
    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch, v="val_")
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        return loss 
    
    def loss(self, batch, v=""):
        x, y = batch[0].float(), batch[1].float()
        y_hat = self(x)
        hr_hat, flux_hat = y_hat[:,:-self.output_len], y_hat[:,-self.output_len:]
        hr = y[:,:-self.output_len]
        flux = y[:,-self.output_len:]
        l = hr_hat.shape[-1]
        qconv = x[:,-l:]
        q_r = hr_hat/86400/qconv # hr_hat in K/s because qconv converts K/s to W/m^2
        fnet_hr = torch.sum(q_r, dim=-1)
        
        mse = F.mse_loss(flux_hat, flux)
        mae = F.l1_loss(flux_hat, flux)
        mbe = torch.abs(torch.mean(flux_hat - flux)) # mean bias error
        self.log(f'{v}mse_loss', mse, sync_dist=True)
        self.log(f'{v}mae_loss', mae, sync_dist=True)
        self.log(f'{v}mse_mae_loss', mse + mae, sync_dist=True)
        self.log(f'{v}mbe', mbe, sync_dist=True)
        if "SW" in self.model_type:
            rsdt = x[:,l*3]  # toa
            alb = x[:,l*3+1] # albedo
            rsds = flux_hat[:,0]
            rsut = flux_hat[:,1]
            #rsus = rsds*alb
            fnet_flux = ((1-rsut)-(1-alb)*rsds)*rsdt # denormalized net-flux
            
            rvds_dir = flux_hat[:, 4]
            rvds_dif = flux_hat[:, 5]
            rnds_dir = flux_hat[:, 6]
            rnds_dif = flux_hat[:, 7]

            sw_loss_dirdif = torch.mean(torch.abs(rvds_dir + rvds_dif + rnds_dir + rnds_dif - rsds))
            self.log(f'{v}sw_loss_dirdif', sw_loss_dirdif, sync_dist=True)
            #mse = mse + sw_loss_dirdif*10
        elif "LW" in self.model_type:
            rlds = flux_hat[:,0]
            rlut = flux_hat[:,1]
            ts_rad = x[:,3*l]
            sig = 5.670374419e-8
            em = 0.996 # emissivity constant in ICON
            rlus = em   #*sig*ts_rad**4
            fnet_flux = (-rlut-(rlds-rlus))*sig*ts_rad**4 # denormalized net-flux
            sw_loss_dirdif = 0
        energy =  torch.mean(torch.abs(fnet_hr - fnet_flux))
        self.log(f'{v}energy_loss', energy, sync_dist=True)
        start_step = 300
        n_steps = 10 
        step = self.current_epoch # self.trainer.global_step
        weight = 0
        if step > start_step:
            weight = np.minimum( 1.e-8*10**((step-start_step)/n_steps), 1.e-1)
            loss = mse + mae + energy * weight
            # increase the weight of the energy loss as the model converges
            # a factor of 10 every 100 epochs
        else:
           loss = mse + mae #+ q_loss
        self.log(f'{v}energy_weight', weight, sync_dist=True)
        return loss

    def on_after_backward(self):
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:  # log every n steps
            for name, params in self.named_parameters():
                if params.grad is not None and torch.any(params.grad):
                    self.logger.experiment.add_histogram(name, params.grad, self.trainer.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,  patience=20, threshold=0.0001, threshold_mode='rel')

        return {"optimizer":optimizer, "lr_scheduler":{"scheduler": scheduler, "monitor": "val_mse_loss"}}
