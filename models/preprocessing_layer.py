
import torch
import torch.nn as nn
import torch.nn.functional as F

class Preprocessing(nn.Module):
    def __init__(self, norm_file, in_vars, mode, pad_len=0, var_len=47):
        super(Preprocessing, self).__init__()
        
        self.norm_file = {key: {"mean": torch.tensor(norm_file[key]["mean"]),
                          "std": torch.tensor(norm_file[key]["std"])}
                    for key in norm_file}
        self.in_vars = in_vars
        self.mode = mode
        self.pad_len = pad_len
        self.var_len = var_len
        assert ["extra_3d_cli", "extra_3d_clw",  "extra_3d_hus"] == self.in_vars[:3] or ["extra_3d_in_cli", "extra_3d_in_clw",  "extra_3d_in_hus"] == self.in_vars[:3]

    def forward(self, x):
        start = 0
        mean = self.norm_file["h2o"]["mean"]            
        std = self.norm_file["h2o"]["std"]     
        l = len(mean)
        cli, clw, hus = x[:, start :start +l], x[:, start +l:start +2*l], x[:, start +2*l:start +3*l]
        h2o = cli + clw + hus
        start += 3*l
        cli = cli / h2o
        clw = clw / h2o
        h2o = (h2o - torch.mean(mean))/ torch.mean(std)
        if self.mode == "horizontal":
            new_x = torch.cat((cli[:,:,None], clw[:,:,None], h2o[:,:,None]), dim=-1)
        elif self.mode == "vertical":
            new_x = torch.cat((cli, clw, h2o), dim=-1)
        else:
            raise ValueError("mode must be vertical or horizontal")
        for v in self.in_vars[3:]:
            if v in ["extra_2d_albedo", "cl", "extra_3d_in_cl", "extra_2d_in_albedo", "albvisdir", "albvisdif", "albnirdir", "albnirdif"]:
                l = 1 if "2d" in v else self.var_len
                d_norm = x[:, start :start +l]
            elif v in ["toa", "toa_hr", "extra_2d_in_toa_flux"]:
                l=1
                d_norm = x[:, start :start +l]/1360
            else:
                #v = var_to_normvar[v]
                mean = self.norm_file[v]["mean"]
                std = self.norm_file[v]["std"]
                l = len(mean) if mean.dim() > 0 else 1
                d = x[:, start :start +l]
                d_norm = (d - torch.mean(mean))/ torch.mean(std)
            
            if self.mode == "horizontal":
                if l<self.pad_len:
                    d_norm = F.pad(d_norm, (self.pad_len-l, 0), mode="replicate")
                new_x = torch.cat((new_x, d_norm[:,:,None]), dim=-1)
            elif self.mode == "vertical":
                new_x = torch.cat((new_x, d_norm), dim=-1)
            else:
                raise ValueError("mode must be vertical or horizontal")
            start += l
        
        return new_x