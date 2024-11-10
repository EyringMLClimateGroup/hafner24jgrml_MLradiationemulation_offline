
import torch
import numpy as np
from preprocessing.load_data import *
import random
import os
from glob import glob
import xarray as xr
from utils import quick_helpers as qh
import pandas as pd
from models.bilstm import BiLSTM
from models.flux import Flux
import yaml
from argparse import Namespace

var_to_normvar = { # this mapping was used because sometimes variables have different names depending on their format e.g. .nc or .grb
    "cli":"cli", 
    "clw":"clw", 
    "cl":"cl",
    "pfull":"pfull", 
    "ta":"ta",
    "hur":"hur",
    "extra_3d_cli":"extra_3d_cli", 
    "extra_3d_clw":"extra_3d_clw", 
    "extra_3d_pfull":"extra_3d_pfull", 
    "extra_3d_ta":"extra_3d_ta",
    "extra_3d_hus":"extra_3d_hus",
    "cosmu0":"cosmu0", 
    "extra_2d_albedo":"extra_2d_albedo",
    "extra_2d_prw":"extra_2d_prw", 
    "extra_2d_clivi":"extra_2d_clivi", 
    "extra_2d_cllvi":"extra_2d_cllvi",
    "rsut":"rsut",
    "rlut":"rlut",
    "rsu":"rsu",
    "rlu":"rlu",
    "rsd":"rsd",
    "rld":"rld",
    "ts_rad": "ts_rad",
    "emissivity": "emissivity",
    "rho": "rho",
    "hfls": "hfls",
    "hfss": "hfss",
    "o3": "o3",
    "rsds": "rsds", 
    "rsus": "rsus",
    "rlds": "rlds", 
    "rlus": "rlus",
    "rvds_dir": "rvds_dir",
    "rvds_dif": "rvds_dif", 
    "rpds_dir": "rpds_dir", 
    "rpds_dif": "rpds_dif", 
    "rnds_dir": "rnds_dir", 
    "rnds_dif": "rnds_dif",
    "tend_ta_rsw": "tend_ta_rsw",
    "tend_ta_rlw": "tend_ta_rlw",
    "dz": "dz",
    "extra_3d_in_cli": "extra_3d_cli", 
    "extra_3d_in_clw": "extra_3d_clw",  
    "extra_3d_in_hus": "extra_3d_hus",
    "extra_2d_in_ts_rad": "extra_2d_ts_rad", 
    "extra_3d_in_cl": "cl", 
    "extra_3d_in_o3": "o3", 
    "extra_3d_in_t": "extra_3d_ta", 
    "extra_3d_in_rho": "rho",
    "extra_2d_in_toa_flux": "toa",
}
   

def create_model(x_shape, y_shape, model_type,  nft, in_vars, extra_shape=47, seed=3141):

    if "FLUX" in model_type:
        model = Flux(model_type, x_shape-extra_shape, nft, in_vars, n_out_nodes=y_shape-extra_shape ,extra_shape=extra_shape)
    elif "HR" in model_type:
        model = BiLSTM(model_type, y_shape - 1, nft, in_vars, extra_shape=extra_shape)
    return model
    
def setup_args_and_load_data(train=True, args_via_commandline=True, *overwrite_args):
    
    if args_via_commandline:
        config_file = qh.arg_parser()
        with open(config_file.config_file, "r") as file:
            args = yaml.safe_load(file)
    else:
        args={}
    if overwrite_args: # for executing the script in a notebook
        for key, val in overwrite_args[0].items():
            print(key,val)
            args[key] = val
    args = Namespace(**args)
    # set random seeds for reproducibility
    np.random.seed(args.seed) # random seed for numpy
    random.seed(args.seed) # random seed for python
    torch.manual_seed(args.seed) # random seed for pytorch

    # Get cpu, gpu or mps device for training.
    args.device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    args.num_gpus = torch.cuda.device_count()
    if args.dev:
        args.folder="test"

    print(f"Using {args.num_gpus} {args.device} device")
    args.save_folder = "results/"+ args.folder +"/"
    args.result_folder = "results/"+ args.folder +f"/{args.model_type}_{args.eval_on}/"
    args.model_path=args.save_folder+f"baseline_{args.model_type}/model.pth"
    args.pretrained_path =  f"results/{args.pretrained}/baseline_{args.model_type}/model.pth"
    args.checkpoint_path = args.save_folder+f"baseline_{args.model_type}/my_checkpoint/"

    os.makedirs(args.result_folder, exist_ok=True)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.save_folder, exist_ok=True)

    args.norm_file = pd.read_pickle(args.norm_file_path)

    # # Data
    types = ["atm3d","atm2d"] 
    exp_base=f"{args.base}{args.exp_name}/{args.exp_name}_"
    grid_path = f"{args.base}{args.exp_name}/icon_grid_0019_R02B05_G.nc"
    bc_solar = f"{args.base}{args.exp_name}/bc_solar_irradiance_sw_b14.nc"
    vgrid_path = f"{exp_base}vgrid_ml.nc"
    args.vgrid = xr.load_dataset(vgrid_path)
    args.grid = xr.load_dataset(grid_path)

    if args.dev: # Number of months+1 to use for training, validation and testing
        n=2
    else:
        n=13
    year = 1979
    train_paths = [np.concatenate([glob(f"{exp_base}{t}_ml_{year}{j:02}0*0000Z.nc") for t in types]) for j in range(1, n)]
    train_paths = np.concatenate(train_paths)

    train_paths = [tp for tp in train_paths if "19790101T000000Z" not in tp]

    val_paths = [np.concatenate([glob(f"{exp_base}{t}_ml_{year}{j:02}1*0000Z.nc") for t in types]) for j in range(1,n,2)]
    val_paths = np.concatenate(val_paths)

    test_paths = [np.concatenate([glob(f"{exp_base}{t}_ml_{year}{j:02}2*0000Z.nc") for t in types]) for j in range(1,n)]
    test_paths = np.concatenate(test_paths)
    #test_paths = [tp for tp in test_paths if "19790101T000000Z" not in tp]

    if train:
        if "LW" in args.model_type:
            step = 10
        else:
            step = 3
        if args.dev:
            step = 10000
        cells = qh.cell_filter_from_data(train_paths, args.model_type, step = step)
        args.coarse_train = CoarseDataGenerator(train_paths, grid_path, bc_solar, vgrid_path, args.variables, var_to_normvar, args.norm_file, x_stack=args.x_mode, y_stack=args.y_mode, cell_filter=cells, shuffle=True, batch_size=args.batch_size)
        
        step = 10 # use less data for validation
        if args.dev:
            step = 10000
        cells = qh.cell_filter_from_data(val_paths, args.model_type, step = step)
        args.coarse_val = CoarseDataGenerator(val_paths, grid_path, bc_solar, vgrid_path, args.variables, var_to_normvar, args.norm_file, x_stack=args.x_mode, y_stack=args.y_mode, cell_filter=cells, shuffle=True, batch_size=args.batch_size)

        print("Getting shapes right...")
        args.train_steps = args.coarse_train.__len__()
        args.validation_steps = args.coarse_val.__len__()

        item = args.coarse_train.__getitem__(0)
        args.extra_shape = args.coarse_train.extra_shape
    
    else:
        args.result_file = args.save_folder+'results_'+ args.model_type + "_" +args.eval_on +'.pickle'
        
        step = 1 # use less data for testing the code
        paths = {"train": train_paths,
                "validation": val_paths,
                "test": test_paths}
        eval_paths=paths[args.eval_on]
        
        cells = qh.cell_filter_from_data(eval_paths, args.model_type, step = step)
        args.coarse_test = CoarseDataGenerator(eval_paths, grid_path, bc_solar, vgrid_path, args.variables, var_to_normvar, args.norm_file, x_stack=args.x_mode, y_stack=args.y_mode, cell_filter=cells, shuffle=False, batch_size=args.batch_size)
        
        item = args.coarse_test.__getitem__(0)
        args.extra_shape = args.coarse_test.extra_shape

    if args.batch_size == 1:
        args.x_shape = item[0].shape[0] # first batch, x, first element in batch
        args.y_shape = item[1].shape[0] # first batch, y, first element in batch
        if args.x_mode == "horizontal":
            args.x_shape = item[0].shape # first batch, x, first element in batch
        if args.y_mode == "horizontal":
            args.y_shape = item[1].shape[-1] # first batch, y, first element in batch

    else:
        args.x_shape = item[0][0].shape[0] # first batch, x, first element in batch
        args.y_shape = item[1][0].shape[0] # first batch, y, first element in batch
        if args.x_mode == "horizontal":
            args.x_shape = item[0][0].shape # first batch, x, first element in batch
        if args.y_mode == "horizontal":
            args.y_shape = item[1][0].shape[-1] # first batch, y, first element in batch

    print("x shape: ", args.x_shape)
    print("y shape: ", args.y_shape)
    print("extra shape: ", args.extra_shape)

    return args


