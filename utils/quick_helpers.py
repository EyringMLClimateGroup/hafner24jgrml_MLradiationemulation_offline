import xarray as xr
import argparse
from glob import glob
import numpy as np
from models.flux import Flux
from models.bilstm import BiLSTM


def var_list(ds):
    """
    returns a list of strings that contains all variable names
    """
    s = "["
    for k in ds.variables:
        s+=f'"{k}", '
    s+="]"
    print(s)
    
def arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c", "--config_file")
    return parser.parse_args()

def has_decimal_point(num):
    return '.' in str(num) and str(num)[-1] != '0'

def cell_filter_from_data(path, model_type, step = 1, filter=True):
    """
    Processes a multi-file dataset and returns a subset of it based on certain conditions.

    Parameters:
    path (str): The path to the multi-file dataset.
    model_type (str): The model type, used to determine the conditions for subsetting the dataset.
    step (int): The step size for reducing the size of the subset arrays.

    Returns:
    list: A list of arrays, where each array is a subset of the 'ncells' field of the dataset. The subsets are determined based on the conditions and the step size.

    """
    import time
    start = time.time()
    paths_2d = [p for p in path if "atm2d" in p]

    data = xr.open_mfdataset(paths_2d, data_vars=["extra_2d_albedo", "cosmu0", "clt"], chunks={"time": 1})
    end = time.time()
    print(  "time to open: ", end-start)
    c = data.ncells.values
    n = len(data.time)
    print(n, c.shape)
    
    start = np.random.randint(0, step, size=n) if step > 1 else [0]*n
    print(start)
    print(c.shape)

    if "SW" in model_type and filter:
        cond1 = np.where(data["extra_2d_albedo"].values>0, 1, 0)
        cond2 = np.where(data["cosmu0"].values>0, 1, 0)
        cond3 = np.where(data["clt"].values>=0, 1, 0)
        cond= (cond1+cond2+cond3)==3
        cells = [c[cond[i]] for i in range(0,n)] # condition filter 
    elif filter:    
        cond1 = np.where(data["clt"].values>=0, 1, 0)
        cond= (cond1)==1
        cells = [c[cond[i]] for i in range(0,n)] # condition filter 
    else:
        cells = c # no condition filter
    
    
    if step > 1:   
        cells = [cells[i][start[i]::step] for i in range(0,n)] # reduction filter
    print([len(c) for c in cells])
    return cells

def load_from_checkpoint(args, extra_shape):
    print("loading checkpoint")
    ckpt = glob(args.checkpoint_path+"*.ckpt")
    if len(ckpt) > 0:
        ckpt.sort() 
    print("Checkpoint: ", ckpt[-1])

    if "FLUX" in args.model_type:
        baseline_model = Flux.load_from_checkpoint(ckpt[-1], 
                                                            model_type=args.model_type, 
                                                            in_nodes=args.x_shape-args.extra_shape, 
                                                            n_out_nodes=args.y_shape-args.extra_shape,
                                                            extra_shape=extra_shape,
                                                            nft=args.nft, 
                                                            in_vars=args.variables["in_vars"])
    elif "HR" in args.model_type:
        baseline_model = BiLSTM.load_from_checkpoint(ckpt[-1],
                                                            model_type=args.model_type,
                                                            output_features=args.y_shape - 1,
                                                            extra_shape=extra_shape,
                                                            nft=args.nft, 
                                                            in_vars=args.variables["in_vars"])
    return baseline_model   


def label_translate(var):
    labels = {"extra_3d_hus": "$q_v$ [km]",
                  "extra_3d_cli": "$q_i$ [km]",
                  "extra_3d_clw": "$q_l$ [km]",
                  "rho": "$\\rho$ [km]",
                  "o3": "O3 [km]",
                  "extra_3d_ta": "T [km]",
                  "extra_2d_albedo": "$\\alpha$",
                  "toa": "$F_{in,TOA}$",
                  "cl": "cloud fr. [km]",
                  "ts_rad": "$T_{S,rad}$",
                  "albnirdir": "$\\alpha_{partial}$",
                  "albnirdif": "",
                  "albvisdir": "",
                  "albvisdif": "",
                  "tend_ta_rsw": "$\partial T_{SW} / \partial t $ [km]",
                  "tend_ta_rlw": "$\partial T_{LW} / \partial t $ [km]",
                  "rsds": "$F_{\downarrow,surf}$ ",
                   "rnds_dir":  "$F_{\downarrow, surf, NIR, dir}$ ",
                   "rnds_dif":  "$F_{\downarrow, surf, NIR, dif}$ ",
                   "rvds_dir":  "$F_{\downarrow,surf,vis,dir}$ ",
                   "rvds_dif": "$F_{\downarrow,surf,vis,dif}$ ",
                   "rpds_dir":  "$F_{\downarrow,surf,PAR,dir}$ ",
                   "rpds_dif":  "$F_{\downarrow,surf,PAR,dif}$",
                   "rsut": "$F_{\\uparrow,TOA}$",
                   "rlds":  "$F_{\downarrow,surf}$",
                   "rlut": "$F_{\\uparrow,TOA}$",
                  
                  }
    return labels[var]