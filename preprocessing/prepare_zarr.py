import xarray as xr
import numpy as np
from glob import glob
import random
from tqdm import tqdm
random.seed(3141)
np.random.seed(3141)

# paths
coarse = "/path/to/data/"
train_paths = np.concatenate([glob(f"{coarse}*2d_ml_1979{j:02}0*0000Z.nc") for j in range(1, 13)])
val_paths = np.concatenate([glob(f"{coarse}*2d_ml_1979{j:02}1*0000Z.nc") for j in range(1, 13)])
test_paths = np.concatenate([glob(f"{coarse}*2d_ml_1979{j:02}2*0000Z.nc") for j in range(1, 13)])

# extract date of form 19790101T000000Z from file name
extract_date = lambda path: path.split("_")[-1].split(".")[0]
train_dates = np.array([extract_date(tp) for tp in train_paths])[1:] # remove first date because it is nonsense 
val_dates = np.array([extract_date(tp) for tp in val_paths])
test_dates = np.array([extract_date(tp) for tp in test_paths])

preprocessed_data_path = "data/"

# variables that need special care: "rlds", "rsdt"
variables_3d_sw = ["o3", "rho", "cl", "tend_ta_rsw", "extra_3d_hus", "extra_3d_cli", "extra_3d_clw", 
                   "qconv", "rsd", "rsu", "extra_3d_pfull", "extra_3d_ta"]
variables_3d_lw = ["o3", "rho", "cl","tend_ta_rlw", "extra_3d_hus", "extra_3d_cli", "extra_3d_clw", 
                   "qconv", "rld", "rlu", "extra_3d_pfull", "extra_3d_ta"]
variables_2d_sw = ["cosmu0", "rsut", "rsds", "rvds_dir", "rvds_dif", "rpds_dir", "rpds_dif", "rnds_dir", 
                   "rnds_dif", "extra_2d_albedo", "extra_2d_clivi", "extra_2d_cllvi", "extra_2d_prw", 
                   "clt", "albvisdir","albvisdif", "albnirdir", "albnirdif", "rsus", "cosmu0_rt"]
variables_2d_lw = ["ts_rad", "rlut", "rlds", "extra_2d_clivi", "extra_2d_cllvi", "extra_2d_prw", 
                   "clt", "rlus","sftlf", "ps", "orog", "sic"]

def create_zarr(dates, variables_3d, variables_2d, sw=False, random_filter=True, partition=""):
    for i, d in enumerate(tqdm(dates)):
        dataset_list = []
        files_3d = glob(f"{coarse}*atm3d*{d}.nc")
        data_3d = xr.open_mfdataset(files_3d)
        data_3d_sel = data_3d[variables_3d]
        if sw:
            var = "toa"
            sel_var = data_3d["rsd"][:,0] # top entry of rsd is TOA
        else:
            var = "rlds_rld" 
            sel_var = data_3d["rld"][:,-1] # last entry of rld is surface
        sel_da = xr.DataArray(
                sel_var, 
                name=var,
                coords=[data_3d.time.values, data_3d.ncells.values],
                dims=["time", "ncells"],
            )
        data_3d.close()
        dataset_list.append(data_3d_sel)
        dataset_list.append(sel_da)

        
        files_2d = glob(f"{coarse}*atm2d*{d}.nc")
        data_2d = xr.open_mfdataset(files_2d)
        data_2d_sel = data_2d[variables_2d]
        data_2d.close()
        dataset_list.append(data_2d_sel)

        merged = xr.merge(dataset_list)
        if sw:
            merged = merged.where(merged["cosmu0"].compute()>0, drop=True)
            mode="SW"
        else:
            mode="LW"
        if partition == "test":
            n_samples = 40000
        elif partition == "val":
            n_samples = 5000
        else:
            n_samples = 14000
        if random_filter:
            # select random cells: make cell_idx (=random_cells) a variable and let cell be np.arange(0, len(random_cells))
            random_cells = np.random.choice(merged.ncells, size=n_samples, replace=False)
            cells_da = xr.DataArray(
                random_cells[np.newaxis,:], 
                name="cell_idx",
                coords=[merged.time, np.arange(0,len(random_cells))],
                dims=["time", "ncells"],
            )  
            random_selection = merged.sel({"ncells": random_cells})
            random_selection["ncells"] = np.arange(0,len(random_cells))
            rd_sel_with_idx = xr.merge([random_selection, cells_da])
        else:
            rd_sel_with_idx = merged        
        
        if i == 0:
            rd_sel_with_idx.to_zarr(f"{preprocessed_data_path}{partition}_{mode}.zarr", mode="w")
        else:
            rd_sel_with_idx.to_zarr(f"{preprocessed_data_path}{partition}_{mode}.zarr", mode="a-", append_dim="time")
        rd_sel_with_idx.close()

create_zarr(train_dates, variables_3d_sw, variables_2d_sw, sw=True, partition="train")
create_zarr(val_dates, variables_3d_sw, variables_2d_sw, sw=True, partition="val")
create_zarr(test_dates, variables_3d_sw, variables_2d_sw, sw=True, partition="test")
create_zarr(train_dates, variables_3d_lw, variables_2d_lw, sw=False, partition="train")
create_zarr(val_dates, variables_3d_lw, variables_2d_lw, sw=False, partition="val")
create_zarr(test_dates, variables_3d_lw, variables_2d_lw, sw=False, random_filter=False, partition="test")


# saving sample data
for subset in ["test_SW", "train_SW", "val_SW", "train_LW", "val_LW", "test_LW", ]:
    ds = xr.open_dataset(f"data/{subset}.zarr", engine="zarr", chunks={}).load()
    ds.isel({"time":[1,2]}).to_zarr(f"test_data/{subset}.zarr")
   