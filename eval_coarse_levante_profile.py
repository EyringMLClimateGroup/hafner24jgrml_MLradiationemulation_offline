import numpy as np
import os
import pickle
from matplotlib.colors import Normalize
import torch
import palettable as pt
import matplotlib as mpl
#Custom imports
from plotter.predict import *
from plotter.shap import *
from plotter.map_plots import plot_map_lat_profile
from evaluation.predict import *
from evaluation.shap import *
import config
from utils.quick_helpers import load_from_checkpoint
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
model = None
args = config.setup_args_and_load_data(train=False)

if "SW" in args.model_type:
    model_name = "Shortwave"
    hr="tend_ta_rsw"
    us="rsus" # upward flux surface
    ds="rsds" # downward flux surface
    ut="rsut" # upward flux toa
elif "LW" in args.model_type:
    model_name = "Longwave"
    hr="tend_ta_rlw"
    us="rlus" # upward flux surface
    ds="rlds" # downward flux surface
    ut="rlut" # upward flux toa


if os.path.exists(args.result_file):
    print("Loading predictions...")
    with open(args.result_file, 'rb') as handle:
        results = pickle.load(handle)
    print("Done!")
else:
    print("Predicting...") 
    model = config.create_model(args, extra_shape=0)
    print(args.save_folder+"baseline_"+ args.model_type +"/model")
    
    try:
        model.load_state_dict(torch.load(args.model_path))
    except:
        model = load_from_checkpoint(args, args.extra_shape)

    results = predict(model.to("cpu"), args.coarse_test, args.norm_file, args.variables, args.model_type, args.y_mode)
    if "HR" in args.model_type:
        results[f"true_{hr}"] = results[f"true_{hr}"]*86400
    with open(args.result_file, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done!")
        

summary_file = args.save_folder+'summary_'+ args.model_type +'.pickle'
if os.path.exists(summary_file):
    print("Loading summary...")
    with open(summary_file, 'rb') as handle:
        summary = pickle.load(handle)
    print("Done!")
else:
    print("Calculating summary...")
    summary = summary_statistics(results, args.grid, args.model_type, args.variables["out_vars"])
    with open(summary_file, 'wb') as handle:
        pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done!")

varlist = [hr, us, ds, ut]
mean_pres = np.mean(results["extra_3d_pfull"], axis=0)
height=np.mean(args.vgrid["zfull"].values, axis=-1)/1000 # in km

if "FLUX" in args.model_type:
    eval_plots(results[f"true_{ds}"], results[f"pred_{ds}"], f"{args.model_type} Downward Surface Flux", folder=args.result_folder)
    eval_plots(results[f"true_{ut}"], results[f"pred_{ut}"], f"{args.model_type} Upward TOA Flux", folder=args.result_folder)
    if "SW" in args.model_type:
        for v in args.variables["out_vars"][2:]:
            eval_plots(results[f"true_{v}"], results[f"pred_{v}"], f"{args.model_type} {v}", folder=args.result_folder)
            
    print("First eval plots done!")

if "HR" in args.model_type:
    
    clear_idx = results["clt"]==0
    cloudy_idx = results["clt"]==1
    def hr_eval(true, pred, hr_id="all"):
        ta_hr = statistics(true, pred)
        if "SW" in args.model_type:
            xlim=(-.1,.5)
        elif "LW" in args.model_type:
            xlim=(-.2,1.)
        for scale in ["linear","log"]:
            plot_statistics_profile(ta_hr, mean_pres, "mae", "MAE Heating Rate [K/d]", scale, xlim=xlim, fname=args.result_folder+hr_id+"_heating_rate_mae_r2_"+scale)
            plot_statistics_profile(ta_hr, mean_pres, "rel_mae_true", "Relative MAE Heating Rate", scale, fname=args.result_folder+hr_id+"_heating_rate_relative_mae_r2_"+scale)
            plot_statistics_profile(ta_hr, mean_pres, "rmse", "RMSE Heating Rate [K/d]", scale, xlim=(-1,3), fname=args.result_folder+hr_id+"_heating_rate_rmse_r2_"+scale)
            plot_statistics_profile_height(ta_hr, height, "mae", "MAE Heating Rate [K/d]", scale, xlim=xlim, fname=args.result_folder+hr_id+"_heating_rate_mae_r2_"+scale)
            plot_statistics_profile_height(ta_hr, height, "rel_mae_true", "Relative MAE Heating Rate", scale, fname=args.result_folder+hr_id+"_heating_rate_relative_mae_r2_"+scale)
            plot_statistics_profile_height(ta_hr, height, "rmse", "RMSE Heating Rate [K/d]", scale, xlim=(-1,3), fname=args.result_folder+hr_id+"_heating_rate_rmse_r2_"+scale)
        
        print("Profile plots done!")
        samples = [int(len(pred)*0.1 ), int(len(pred)*0.3 ), int(len(pred)*0.5 ), int(len(pred)*0.6 ), int(len(pred)*0.7 )]
        os.makedirs(args.result_folder+"samples/", exist_ok=True)
        scale="linear"
        plot_sample(true, pred, mean_pres, samples, "Heating Rate [K/d]", scale, fname=args.result_folder+"samples/"+hr_id+"_hr")
        
        return ta_hr
    ## all
    
    ta_hr = hr_eval(results[f"true_{hr}"],  results[f"pred_{hr}"], "all")

    ## clear sky
    _ = hr_eval(results[f"true_{hr}"][clear_idx],  results[f"pred_{hr}"][clear_idx], "clear")

    ## cloudy sky
    _ = hr_eval(results[f"true_{hr}"][cloudy_idx],  results[f"pred_{hr}"][cloudy_idx], "cloudy")
    print("r2 minimum at index: ", np.argmin(ta_hr["r2"]))
    print("Sample plots done!")

    results_clear = {}
    results_cloudy = {}
    for k in results.keys():
        print(k, results[k].shape)
        try:
            results_clear[k] = results[k][results["clt"]==0]
            results_cloudy[k] = results[k][results["clt"]==1]
        except:
            print("failed for ", k)
            continue

    print("All")    
    res = ["all", "clear", "cloudy"]
    title = "Shortwave" if "SW" in args.model_type else "Longwave"
    h_per_x = args.vgrid["zfull"].values/1000
    for r_idx, r in enumerate([results, results_clear, results_cloudy]):
        print(res[r_idx])
        for m in ["MAE", "Bias", "R^2", "o3", "extra_3d_ta", "rho"]:
            if m == "MAE":
                vals = np.abs(r[f"true_{hr}"]-r[f"pred_{hr}"])
                cmap =  mpl.colors.LinearSegmentedColormap.from_list(
                        'Custom cmap', pt.colorbrewer.sequential.YlGnBu_5.mpl_colors, 5)#"YlGnBu"
                norm=Normalize(vmin=0,vmax=0.5)
                summary_f = np.mean
                extend="max"
            elif m == "Bias":
                vals = r[f"true_{hr}"]-r[f"pred_{hr}"]
                cmap = mpl.colors.LinearSegmentedColormap.from_list(
                        'Custom cmap', pt.colorbrewer.diverging.RdBu_5.mpl_colors[::-1], 5)
                norm=Normalize(vmin=-.025,vmax=0.025)
                summary_f = np.mean
                extend="both"
            elif m == "R^2":
                vals = np.hstack((r[f"true_{hr}"][:,np.newaxis,:], r[f"pred_{hr}"][:,np.newaxis,:]))
                cmap = mpl.colors.LinearSegmentedColormap.from_list(
                        'Custom cmap', pt.colorbrewer.sequential.YlGnBu_5.mpl_colors, 5)
                
                def r2(data, axis=0):
                    """
                    data shape (n, 2)
                    """
                    from sklearn.metrics import r2_score
                    ytrue = data[:,0]
                    ypred = data[:,1]
                    r2 = r2_score(ytrue, ypred, multioutput="raw_values")
                    return r2 
                norm=Normalize(vmin=0.8,vmax=1)
                summary_f = r2
                extend="min"
            elif m in r.keys():
                vals = r[m]
                summary_f = np.mean
                norm=Normalize(vmin=np.min(vals),vmax=np.max(vals))
                summary_f = np.mean
                extend="neither"
                cmap =  mpl.colors.LinearSegmentedColormap.from_list(
                        'Custom cmap', pt.colorbrewer.sequential.YlGnBu_5.mpl_colors, 5)
            else:
                print("Don't know mode ", m)
                break
            for s in ["log"]: # ["linear", "log"]
                pres_vs_lat(height, h_per_x[:,r["idx"]], args.grid["clat"].values[r["idx"]], vals, 
                            label=f"{m} Heating Rate [K/d]", norm=norm, extend=extend, ymode="height", title=f"{title}: {res[r_idx]}-sky",
                            fname=f"{args.result_folder}height_lat_{m}_{res[r_idx]}_{s}", scale=s, cmap=cmap, summary_f=summary_f)
            
    if "SW" in args.model_type:
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, np.mean(summary[f"mae_{hr}"], axis=1), 
                                    args.model_type, label="MAE [$K/d$]", norm=Normalize(0.06, vmax=.2, clip=True), extend="both", 
                                    fname=f"{args.result_folder}map_avg_mae_hr")
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, np.mean(summary[f"bias_{hr}"], axis=1), 
                                    args.model_type, label="Bias [$K/d$]", norm=Normalize(-0.05, vmax=.05, clip=True), extend="both",cmap="seismic",
                                    fname=f"{args.result_folder}map_avg_bias_hr" )
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, summary[f"mae_{hr}"][:,-1], 
                                "Near surface"+args.model_type, label="MAE [$K/d$]", norm=Normalize(0., vmax=.6, clip=True), extend="both", 
                                fname=f"{args.result_folder}map_avg_mae_hr_near_surface")
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, summary[f"bias_{hr}"][:,-1], 
                                "Near surface"+args.model_type, label="Bias [$K/d$]", norm=Normalize(-0.3, vmax=.3, clip=True), extend="both",cmap="seismic", 
                                fname=f"{args.result_folder}map_avg_bias_hr_near_surface")
    elif "LW" in args.model_type:
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, np.mean(summary[f"mae_{hr}"], axis=1), 
                                    args.model_type, label="MAE [$K/d$]", norm=Normalize(0.1, vmax=.25, clip=True), extend="both", 
                                    fname=f"{args.result_folder}map_avg_mae_hr")
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, np.mean(summary[f"bias_{hr}"], axis=1), 
                                    args.model_type, label="Bias [$K/d$]", norm=Normalize(-0.02, vmax=.02, clip=True), extend="both",cmap="seismic",
                                    fname=f"{args.result_folder}map_avg_bias_hr" )
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, summary[f"mae_{hr}"][:,-1], 
                                "Near surface"+args.model_type, label="MAE [$K/d$]", norm=Normalize(0.25, vmax=1.8, clip=True), extend="both", 
                                fname=f"{args.result_folder}map_avg_mae_hr_near_surface")
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, summary[f"bias_{hr}"][:,-1], 
                                "Near surface"+args.model_type, label="Bias [$K/d$]", norm=Normalize(-1, vmax=1, clip=True), extend="both",cmap="seismic" ,
                                fname=f"{args.result_folder}map_avg_bias_hr_near_surface")


if "FLUX" in args.model_type:
    vnames = [v for v in summary.keys() if "r2" in v]
    for v in vnames:
        if len(summary[v][0])>1:
            continue
        m = np.mean(summary[v])
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, summary[v], 
                                f"{v} mean: {m:.2f}", label="R2",
                                norm=Normalize(0, vmax=1, clip=True), extend="min",
                                fname=f"{args.result_folder}{v}")

if  "LW_FLUX" in args.model_type:
    vnames = [v for v in summary.keys() if "mae" in v]
    for v in vnames:
        if len(summary[v][0])>1:
            continue
        m = np.mean(summary[v])
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, summary[v], 
                                f"{v} mean: {m:.2f}", label="MAE [$W/m^2$]",
                                norm=Normalize(0, vmax=10, clip=True), extend="max",
                                fname=f"{args.result_folder}{v}")
    vnames = [v for v in summary.keys() if "rel" in v]
    for v in vnames:
        if len(summary[v][0])>1:
            continue
        m = np.mean(summary[v])
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, summary[v], 
                                f"{v} mean: {m:.2f}", label="Relative error",
                                norm=Normalize(0, vmax=0.05, clip=True), extend="max",
                                fname=f"{args.result_folder}{v}")
    vnames = [v for v in summary.keys() if "bias" in v]
    for v in vnames:
        if len(summary[v][0])>1:
            continue
        m = np.mean(summary[v])
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, summary[v], 
                                f"{v} mean: {m:.2f}", label="Bias [$W/m^2$]",cmap="seismic", 
                                norm=Normalize(-5, vmax=5, clip=True), extend="both",
                                fname=f"{args.result_folder}{v}")
elif "SW_FLUX" in args.model_type:
    vnames = [v for v in summary.keys() if "mae" in v]
    for v in vnames:
        if len(summary[v][0])>1:
            continue
        m = np.mean(summary[v])
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, summary[v], 
                                f"{v} mean: {m:.2f}", label="MAE [$W/m^2$]",
                                norm=Normalize(0, vmax=15, clip=True), extend="max",
                                fname=f"{args.result_folder}{v}")
    vnames = [v for v in summary.keys() if "rel" in v]
    for v in vnames:
        if len(summary[v][0])>1:
            continue
        m = np.mean(summary[v])
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, summary[v], 
                                f"{v} mean: {m:.2f}", label="Relative error",
                                norm=Normalize(0, vmax=0.5, clip=True), extend="max",
                                fname=f"{args.result_folder}{v}")
    vnames = [v for v in summary.keys() if "bias" in v]
    for v in vnames:
        if len(summary[v][0])>1:
            continue
        m = np.mean(summary[v])
        plot_map_lat_profile(summary["clat"].values, summary["clon"].values, summary[v], 
                                f"{v} mean: {m:.2f}", label="Bias [$W/m^2$]",cmap="seismic", 
                                norm=Normalize(-10, vmax=10, clip=True), extend="both",
                                fname=f"{args.result_folder}{v}")

print("Map plots done!")

print("Calculating energy consistency...")
if "SW" in args.model_type:
    dt = results["toa"]
    net_surf = results[f"pred_{ds}"]*(1-results["extra_2d_albedo"])
else:
    dt = 0
    sig= 5.670374419e-8
    net_surf = results[f"pred_{ds}"] - 0.996*results["ts_rad"]**4*sig

net_toa = dt - results[f"pred_{ut}"] 
fnet_flux = net_toa - net_surf
fnet_hr = results[f"pred_{hr}"]/86400/results["qconv"]
diff = fnet_flux - np.sum(fnet_hr, axis=-1)

p025 = np.percentile(diff, 1)
p975 = np.percentile(diff, 99)
m = np.mean(diff)
am = np.mean(np.abs(diff))
s = np.std(diff)
xlim = np.max(np.abs([p025,p975]))
print(m, am, s, p025, p975)
text = f" $\\mu$  =   {m:.2f} $W/m^2$ \n$|\mu|$ = {am:.2f} $W/m^2$ \n $\sigma$  = {s:.2f} $W/m^2$"
print(text)

plt.figure()
plt.title(f"Energy Consistency - {model_name}")
import matplotlib as mpl
mpl.rcParams['font.size'] = '15'
plt.hist(diff, bins=100, density=True, range=(-xlim, xlim))
plt.ylabel("Density")
plt.xlabel("Difference [$W/m^2$]")
plt.text(10,0.015,text, backgroundcolor="w")
plt.tight_layout()
plt.savefig(args.result_folder+"energy_consistency.png")
plt.close()


print("Starting SHAP analysis...")
if os.path.exists(args.result_folder+"shapley_mean_vals.npy"):
    print("Loading SHAP values...")
    shapley_mean_vals = np.load(args.result_folder+"shapley_mean_vals.npy")
else:
    print("Calculating SHAP values...")
    model = config.create_model(args, shap =True)
    model.load_state_dict(torch.load(args.model_path))
    shapley_mean_vals = calculate_mean_shapley(model, args)
    np.save(args.result_folder+"shapley_mean_vals.npy", shapley_mean_vals)
plot_interpret_values(shapley_mean_vals, height, args.variables, args.norm_file, args.model_type)