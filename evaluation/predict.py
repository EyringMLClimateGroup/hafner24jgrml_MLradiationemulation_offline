import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score
from datetime import datetime
from pymeeus.Epoch import Epoch
from pymeeus.Earth import Earth
import torch


def denorm(var, varname, nf="", norm_type="minmax", batch=""):
    """
    Denormalizes the given variable based on the specified normalization type.
    Parameters:
    - var: The variable to be denormalized.
    - varname: The name of the variable.
    - nf: The normalization factors (optional).
    - norm_type: The type of normalization to be applied (default: "minmax").
    - batch: The batch data (optional).
    Returns:
    - The denormalized variable.
    """
    if norm_type=="minmax":
        min_val = nf[varname]["min"]
        max_val = nf[varname]["max"]
        diff = max_val - min_val
        denorm = np.where(diff>0, var*diff+min_val, min_val)
    elif norm_type=="mean_minmax":
        min_val = nf[varname]["min"]
        max_val = nf[varname]["max"]
        diff = max_val - min_val
        mean_val = nf[varname]["mean"]
        denorm = np.where(diff>0, var*diff+mean_val, mean_val)
    elif norm_type=="SW":
        sw_norm = batch["rsd"][0].values
        denorm = var*sw_norm            
    elif norm_type=="LW":
        sig = 5.670374419e-8
        ta = batch["ts_rad"].values
        denorm = var*(ta**4*sig)
    elif norm_type=="exp":
        denorm = var**4
    return denorm

def get_special_var(v, data_gen, t, batch_idx, cell_name):
    """
    Retrieves a special variable based on the given parameters.
    Parameters:
    - v (str): The variable to retrieve.
    - data_gen (object): The data generator object.
    - t (int): The time index.
    - batch_idx (int): The batch index.
    - cell_name (str): The cell name.
    Returns:
    - The value of the requested variable.
    Raises:
    - NotImplementedError: If the requested variable is not supported.
    """
    match v:
        case "toa":
            return data_gen.data["rsd"].isel({"time": t,cell_name: batch_idx})[0]
        case "toa_hr":
            cur_time = data_gen.data.time[t].values
            date = datetime.strptime(str(np.round(cur_time, 3)), "%Y%m%d.%f")
            tsi = data_gen.bc_solar.TSI[(data_gen.bc_solar.year==date.year)& (data_gen.bc_solar.month==date.month)].values
            this_date = Epoch(date)
            _,_,dist = Earth().geometric_heliocentric_position_j2000(this_date) 
            
            d = data_gen.data["cosmu0"].isel({"time": t,cell_name: batch_idx}).values*tsi/dist**2
            return d.squeeze()
        case "rsus":
            return data_gen.data["rsus"].isel({"time": t,cell_name: batch_idx}).squeeze()
        case "rsut":
            return data_gen.data["rsu"].isel({"time": t,cell_name: batch_idx})[0]
        case "rsds":
            return data_gen.data["rsds"].isel({"time": t,cell_name: batch_idx}).squeeze()
        case "rlus":
            return data_gen.data["rlu"].isel({"time": t,cell_name: batch_idx})[-1]
        case "rlut":
            return data_gen.data["rlu"].isel({"time": t,cell_name: batch_idx})[0]
        case "rlds":
            return data_gen.data["rld"].isel({"time": t,cell_name: batch_idx})[-1]
        case "h2o":
            h2o = data_gen.data["extra_3d_cli"].isel({"time": t,cell_name: batch_idx}) + data_gen.data["extra_3d_clw"].isel({"time": t,cell_name: batch_idx}) + data_gen.data["extra_3d_hus"].isel({"time": t,cell_name: batch_idx})
            return  h2o
        case "dz":
            return data_gen.vgrid["dzghalf"].isel({cell_name: batch_idx})
        case _:
            NotImplementedError()

def predict(model, data_gen, nf, variables, model_type, mode="vertical"):
    """
    Predicts the output using the given model and data generator.
    Parameters:
    - model: The trained model used for prediction.
    - data_gen: The data generator object that generates input data for prediction.
    - nf: The normalization factor used for denormalization.
    - variables: A dictionary containing the input and output variables.
    - model_type: The type of the model used for prediction.
    - mode: The mode of prediction, either "vertical" or "horizontal".
    Returns:
    - return_dict: A dictionary containing the predicted and true values for each variable.
    """
    special_vars = ["q", "toa", "toa_hr", "rsus", "rsut", "rsds", "rlus", "rlut", "rlds","h2o", "dz"]
    x, true = data_gen.__getitem__(0)
    cell_name = data_gen.cell
    if data_gen.batch_size is None:
        t = 0
        batch_idx = data_gen.cells[t]
    else:
        t,c = data_gen.idx_to_tc(0)
        batch_idx = data_gen.cells[t][c:c+data_gen.batch_size]
    
    return_dict = {
        "idx": batch_idx,
        "t": [t]*len(batch_idx)
    }
    
    for v in variables["in_vars"] + variables["diagnostics"]:
        if v in special_vars:
            temp_var = get_special_var(v, data_gen, t, batch_idx, cell_name)                
        else:
            temp_var = data_gen.data[v].isel({"time": t,cell_name: batch_idx}).squeeze()
        s = temp_var.shape    
        print(v, s)
        if len(s)==1:
            return_dict[v] = temp_var
        else:
            return_dict[v] = temp_var.T
    pred = model.predict(torch.from_numpy(x)).detach().numpy()

    if "SW" in model_type:
        hr="tend_ta_rsw"
        us="rsus" # upward flux surface
        ds="rsds" # downward flux surface
        ut="rsut" # upward flux toa
        nt="SW"   # norm type for denorming fluxes
    elif "LW" in model_type:
        hr="tend_ta_rlw"
        us="rlus" # upward flux surface
        ds="rlds" # downward flux surface
        ut="rlut" # upward flux toa
        nt="LW"   # norm type for denorming fluxes
    if "FLUX_HR" in model_type:
        return_dict[f"true_{hr}"] = data_gen.data[hr].isel({"time": t,cell_name: batch_idx}).T
        return_dict[f"pred_{hr}"] = pred[:,:47]
        return_dict[f"true_{ds}"] = get_special_var(ds, data_gen, t, batch_idx, cell_name)
        return_dict[f"true_{ut}"] = get_special_var(ut, data_gen, t, batch_idx, cell_name)
        return_dict[f"pred_{ds}"] = denorm(pred[:,47], ds, norm_type=nt, batch=data_gen.data.isel({"time": t,cell_name: batch_idx}))        
        return_dict[f"pred_{ut}"] = denorm(pred[:,47+1], ut, norm_type=nt, batch=data_gen.data.isel({"time": t,cell_name: batch_idx}))
        if "SW" in model_type:
            for idx, v in enumerate(variables["out_vars"][3:]):
                return_dict[f"true_{v}"] = data_gen.data[v].isel({"time": t,cell_name: batch_idx}).values.squeeze()
                return_dict[f"pred_{v}"] = denorm(pred[:,47+idx+2], v, norm_type="SW", batch=data_gen.data.isel({"time": t,cell_name: batch_idx}))
    elif "HR" in model_type:
        return_dict[f"true_{hr}"] = data_gen.data[hr].isel({"time": t,cell_name: batch_idx}).T
        if mode == "vertical":
            return_dict[f"pred_{hr}"] = pred[:,:47]
        elif mode == "horizontal":
            return_dict[f"pred_{hr}"] = pred[:,:,0]
    elif "FLUX" in model_type:
        return_dict[f"true_{ds}"] = get_special_var(ds, data_gen, t, batch_idx, cell_name)
        return_dict[f"true_{ut}"] = get_special_var(ut, data_gen, t, batch_idx, cell_name)
        return_dict[f"pred_{ds}"] = denorm(pred[:,0], ds, norm_type=nt, batch=data_gen.data.isel({"time": t,cell_name: batch_idx}))        
        return_dict[f"pred_{ut}"] = denorm(pred[:,1], ut, norm_type=nt, batch=data_gen.data.isel({"time": t,cell_name: batch_idx}))
        if "SW" in model_type:
            for idx, v in enumerate(variables["out_vars"][2:]):
                return_dict[f"true_{v}"] = data_gen.data[v].isel({"time": t,cell_name: batch_idx}).values.squeeze()
                return_dict[f"pred_{v}"] = denorm(pred[:,idx+2], v, norm_type="SW", batch=data_gen.data.isel({"time": t,cell_name: batch_idx}))       

    for i in tqdm(range(1, data_gen.__len__())):
        x, y = data_gen.__getitem__(i)
        y_pred =  model.predict(torch.from_numpy(x)).detach().numpy()
        if data_gen.batch_size is None:
            t = i
            batch_idx = data_gen.cells[t]
        else:
            t,c = data_gen.idx_to_tc(i)
            batch_idx = data_gen.cells[t][c:c+data_gen.batch_size]
        return_dict["idx"] = np.hstack((return_dict["idx"], batch_idx))
        return_dict["t"] = np.hstack((return_dict["t"], [t]*len(batch_idx)))
        for v in variables["in_vars"] + variables["diagnostics"]:
            if v in special_vars:
                temp_var = get_special_var(v, data_gen, t, batch_idx, cell_name)                
            else:
                temp_var = data_gen.data[v].isel({"time": t,cell_name: batch_idx}).squeeze()
            s = temp_var.squeeze().shape            
            if len(s)==1:
                return_dict[v] = np.hstack((return_dict[v], temp_var))
            else:
                return_dict[v] = np.vstack((return_dict[v], temp_var.T))

        if "FLUX_HR" in model_type:
            return_dict[f"true_{hr}"] = np.vstack((return_dict[f"true_{hr}"], data_gen.data[hr].isel({"time": t,cell_name: batch_idx}).T))
            return_dict[f"pred_{hr}"] = np.vstack((return_dict[f"pred_{hr}"], y_pred[:,:47]))
            return_dict[f"true_{ds}"] = np.hstack((return_dict[f"true_{ds}"], get_special_var(ds, data_gen, t, batch_idx, cell_name)))
            return_dict[f"true_{ut}"] = np.hstack((return_dict[f"true_{ut}"], get_special_var(ut, data_gen, t, batch_idx, cell_name)))
            return_dict[f"pred_{ds}"] = np.hstack((return_dict[f"pred_{ds}"], denorm(y_pred[:,47], ds, norm_type=nt, batch=data_gen.data.isel({"time": t,cell_name: batch_idx})) ))        
            return_dict[f"pred_{ut}"] = np.hstack((return_dict[f"pred_{ut}"], denorm(y_pred[:,47+1], ut, norm_type=nt, batch=data_gen.data.isel({"time": t,cell_name: batch_idx})) ))
            if "SW" in model_type:
                for idx, v in enumerate(variables["out_vars"][3:]):
                    return_dict[f"true_{v}"] = np.hstack((return_dict[f"true_{v}"], data_gen.data[v].isel({"time": t,cell_name: batch_idx}).values.squeeze()))
                    return_dict[f"pred_{v}"] = np.hstack((return_dict[f"pred_{v}"], denorm(y_pred[:,47+idx+2], v, norm_type="SW", batch=data_gen.data.isel({"time": t,cell_name: batch_idx}))))
        elif "HR" in model_type:
            return_dict[f"true_{hr}"] = np.vstack((return_dict[f"true_{hr}"], data_gen.data[hr].isel({"time": t,cell_name: batch_idx}).T))
            if mode == "vertical":
                return_dict[f"pred_{hr}"] = np.vstack((return_dict[f"pred_{hr}"], y_pred[:,:47]))
            elif mode == "horizontal":
                return_dict[f"pred_{hr}"] = np.vstack((return_dict[f"pred_{hr}"], y_pred[:,:,0]))
        elif "FLUX" in model_type:
            return_dict[f"true_{ds}"] = np.hstack((return_dict[f"true_{ds}"], get_special_var(ds, data_gen, t, batch_idx, cell_name)))
            return_dict[f"true_{ut}"] = np.hstack((return_dict[f"true_{ut}"], get_special_var(ut, data_gen, t, batch_idx, cell_name)))
            return_dict[f"pred_{ds}"] = np.hstack((return_dict[f"pred_{ds}"], denorm(y_pred[:,0], ds, norm_type=nt, batch=data_gen.data.isel({"time": t,cell_name: batch_idx})) ))        
            return_dict[f"pred_{ut}"] = np.hstack((return_dict[f"pred_{ut}"], denorm(y_pred[:,1], ut, norm_type=nt, batch=data_gen.data.isel({"time": t,cell_name: batch_idx})) ))
            if "SW" in model_type:
                for idx, v in enumerate(variables["out_vars"][2:]):
                    return_dict[f"true_{v}"] = np.hstack((return_dict[f"true_{v}"], data_gen.data[v].isel({"time": t,cell_name: batch_idx}).values.squeeze()))
                    return_dict[f"pred_{v}"] = np.hstack((return_dict[f"pred_{v}"], denorm(y_pred[:,idx+2], v, norm_type="SW", batch=data_gen.data.isel({"time": t,cell_name: batch_idx}))))

    return return_dict

def statistics(y_true, y_pred):
    """
    Calculate various statistical metrics to evaluate the performance of a prediction model.
    Parameters:
    - y_true (array-like): True values of the target variable.
    - y_pred (array-like): Predicted values of the target variable.
    Returns:
    - dict: A dictionary containing statistical metrics such as mean absolute error, mean squared error, and R^2 score.
    """

    x = np.abs(y_true - y_pred)
    mae = np.mean(x, axis=0)
    std_m = np.std(x, axis=0, where=x<mae)
    std_p = np.std(x, axis=0, where=x>mae)
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    rel_x_pred = np.where(np.abs(y_true)>0, x/np.abs(y_true), 0)
    rel_x_true = np.where(np.abs(y_pred)>0, x/np.abs(y_pred), 0)


    return {"mae": mae,
            "rel_mae_true": np.mean(rel_x_true, axis=0),
            "rel_mae_pred": np.mean(rel_x_pred, axis=0),
            "rel_true_perc_5": np.percentile(rel_x_true, 5, axis=0),
            "rel_pred_perc_5": np.percentile(rel_x_pred, 5, axis=0),
            "rel_true_perc_95": np.percentile(rel_x_true, 95, axis=0),
            "rel_pred_perc_95": np.percentile(rel_x_pred, 95, axis=0),
            "true": np.mean(y_true, axis=0),
            "pred": np.mean(y_pred, axis=0),
            "mean": np.mean(y_true - y_pred, axis=0),
            "mean_err": np.mean(y_true - y_pred, axis=0),
            "mse": np.mean(x**2, axis=0),
            "rmse": np.sqrt(np.mean(x**2, axis=0)),
            "abs_median": np.median(x, axis=0),
            "median": np.median(y_true - y_pred, axis=0),
            "std_p": std_p,
            "std_m": std_m,
            "min": np.min(y_true - y_pred, axis=0),
            "max": np.max(y_true - y_pred, axis=0),
            "max_abs": np.max(x, axis=0),
            "abs_perc_5": np.percentile(x, 5, axis=0),
            "abs_perc_95": np.percentile(x, 95, axis=0),
            "perc_5": np.percentile(y_true - y_pred, 5, axis=0),
            "perc_95": np.percentile(y_true - y_pred, 95, axis=0),
            "r2": r2,
            "r2_mean": np.mean(r2),
           }


def summary_statistics(result_dict, grid, model_type, varlist, verbose=False):
    """
    Calculate summary statistics for a given result dictionary.
    Parameters:
    - result_dict (dict): A dictionary containing the results.
    - grid (dict): A dictionary containing the grid information.
    - model_type (str): The type of the model.
    - varlist (list): A list of variables for which the variables will be calculated.
    - verbose (bool, optional): Whether to print verbose output. Defaults to False.
    Returns:
    - summary_dict (dict): A dictionary containing the summary statistics.
    """

    clat = grid["clat"]
    clon = grid["clon"]
    n = len(clat)

    summary_dict = {
        "clat": clat,
        "clon": clon}

    for v in varlist:
        if v == "rlds_rld":
            v = "rlds"
        if "tend" in v:
            summary_dict[f"mae_{v}"] = np.zeros((n, 47))
            summary_dict[f"bias_{v}"] = np.zeros((n, 47))
            summary_dict[f"r2_{v}"] = np.zeros((n, 47))
            summary_dict[f"true_{v}"] = np.zeros((n, 47))
            summary_dict[f"pred_{v}"] = np.zeros((n, 47))
            summary_dict[f"rel_{v}"] = np.zeros((n, 47))
        else:
            summary_dict[f"mae_{v}"] = np.zeros((n, 1))
            summary_dict[f"bias_{v}"] = np.zeros((n, 1))
            summary_dict[f"r2_{v}"] = np.zeros((n, 1))
            summary_dict[f"true_{v}"] = np.zeros((n, 1))
            summary_dict[f"pred_{v}"] = np.zeros((n, 1))
            summary_dict[f"rel_{v}"] = np.zeros((n, 1))
    f = 0
    for i in tqdm(range(n)):
        idx = np.argwhere(result_dict["idx"]==i).squeeze()
        
        for v in varlist:
            if v == "rlds_rld":
                v = "rlds"
            if idx.size == 0:
                if verbose:
                    print("Failed to produce summary for index ", i)
                f += 1
                continue
            summary_dict[f"mae_{v}"][i] = np.mean(np.abs(result_dict[f"true_{v}"][idx] - result_dict[f"pred_{v}"][idx]), axis=0)
            summary_dict[f"bias_{v}"][i] = np.mean((result_dict[f"true_{v}"][idx] - result_dict[f"pred_{v}"][idx]), axis=0)
            summary_dict[f"r2_{v}"][i] = r2_score(result_dict[f"true_{v}"][idx], result_dict[f"pred_{v}"][idx], multioutput="raw_values")
            summary_dict[f"true_{v}"][i] = np.mean(result_dict[f"true_{v}"][idx], axis=0)
            summary_dict[f"pred_{v}"][i] = np.mean(result_dict[f"pred_{v}"][idx], axis=0)
            summary_dict[f"rel_{v}"][i] = np.mean(np.abs(np.divide(result_dict[f"true_{v}"][idx] - result_dict[f"pred_{v}"][idx], np.abs(result_dict[f"true_{v}"][idx]), where=np.abs(result_dict[f"true_{v}"][idx])>1e-2)), axis=0)
    
    print(f"Failed to produce summary for {f} indices.")
    return summary_dict

def summary_statistics_var(result_dict, vname, grid):
    """
    Calculate the summary statistics for a given variable in a result dictionary.
    Parameters:
    - result_dict (dict): A dictionary containing the results.
    - vname (str): The name of the variable to calculate the summary statistics for.
    - grid (dict): A dictionary containing the grid information.
    Returns:
    - summary_dict (dict): A dictionary containing the summary statistics.
        - "clat" (array-like): The latitude values of the grid.
        - "clon" (array-like): The longitude values of the grid.
        - f"mean_{vname}" (array-like): The mean values of the variable for each grid point.
    """

    clat = grid["clat"]
    clon = grid["clon"]
    n = len(clat)
    var = result_dict[vname]
    s = var.shape[-1] if len(var.shape) == 2 else 1

    summary_dict = {
        "clat": clat,
        "clon": clon,
        f"mean_{vname}": np.zeros((n, s)),
    }
    for i in tqdm(range(n)):
        idx = np.argwhere(result_dict["idx"]==i).squeeze()
        summary_dict[f"mean_{vname}"][i] = np.mean(var[idx], axis=0)

    return summary_dict