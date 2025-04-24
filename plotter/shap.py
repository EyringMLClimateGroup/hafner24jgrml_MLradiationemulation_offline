import numpy as np
import matplotlib.pyplot as plt
from utils.quick_helpers import label_translate 
import matplotlib as mpl

def get_start_end_in_list(var_list, norm_file, height):
    """
    height: 1, 10, 50 km
    index: 42, 28, 7
    """
    start, end, ticks, labels = [], [], [], []
    for v in var_list[::-1]:
        if not start:
            start.append(0)
        else:
            start.append(end[-1])
        try:
            l = len(norm_file[v]["mean"])
        except:
            if v in ["h2o", "cl", "q"]:
                l=len(norm_file["extra_3d_ta"]["mean"])
            else:
                l = 1
        if l>1:
            i_1 = np.argmin(np.abs(height-1))
            i_10 = np.argmin(np.abs(height-10))
            i_50 = np.argmin(np.abs(height-50))
            labels +=[1,10, 50]
            ticks.append(start[-1]+l-i_1)
            ticks.append(start[-1]+l-i_10)
            ticks.append(start[-1]+l-i_50)
        else: 
            ticks.append(start[-1]+0.5)
            labels.append("")
        end.append(start[-1]+l)
    return np.array(start), np.array(end), np.array(ticks), labels  

model_specs = {
    "LW_FLUX": {
        "mtype": "LW_FLUX",
        #"folder": "fluxes_new",
        #"ymode": "vertical",
        "yxshift": -10,
        "yyshift": 0.8,
        "yshift": 0.25,
        "yrotate": 90,
        "vmin": 0.001,
    },
    "SW_FLUX": {
        "mtype": "SW_FLUX",
        #"folder": "fluxes_mbe",
        #"ymode": "vertical",
        "yxshift": -40,
        "yyshift": 0.5,
        "yshift": 2.5,
        "yrotate": 0,
        "vmin": 0.01,
    },
    "LW_HR": {
        "mtype": "LW_HR",
        #"folder": "preprocessing_toa",
        #"ymode": "horizontal",
        "yxshift": -15,
        "yyshift": 20,
        "yshift": 13,
        "yrotate": 90,
        "vmin": 0.5,
    },
    "SW_HR": {
        "mtype": "SW_HR",
        #"folder": "preprocessing_toa",
        #"ymode": "horizontal",
        "yxshift": -15,
        "yyshift": 20,
        "yshift": 13,
        "yrotate": 90,
        "vmin": 0.3,
    },
    "SW_FLUX_HR": {
        "xx_yy": {
            "figsize": (42, 7),
            "xshift":-15,
            "yshift": -15,
            "vmax": 1,
            "vmin":0.3,
        },
        "xx_y": {
            "figsize": (42,7),
            "yshift": -2,
            "vmax": 1e-1,
            "vmin":5e-2,
        },
        "x_yy": {
            "figsize": (9,7),
            "xshift": -2,
            "vmax": 1,
            "vmin": 5e-1,
        },
        "x_y": {
            "figsize": (5,5),
            "vmax": 5e-2,
            "vmin": 5e-3,
        },
    },
    "LW_FLUX_HR": {
        "xx_yy": {
            "figsize": (42, 7),
            "xshift":-15,
            "yshift": -15,
            "vmax": 1e-13,#1.5,
            "vmin": 1e-20, #0.5,
        },
        "xx_y": {
            "figsize": (42,5),
            "yshift": -1,
            "vmax": 1e-13,#1e-1,
            "vmin": 1e-20,#1e-2,
        },
        "x_yy": {
            "figsize": (2.5,5),
            "xshift": -1,
            "vmax": 1e-13,#2,
            "vmin": 1e-20,#5e-1,
        },
        "x_y": {
            "figsize": (2.5,5),
            "vmax": 1e-13,#3,
            "vmin": 1e-20,#5e-1,
        },
    }
}
def plot_interpret_values(vals, height, variables, norm_file, model_type, **kwargs):
    fig_specs = model_specs[model_type]
    mpl.rcParams['font.size'] = '36'
    h = len(height)
    i_1 = np.argmin(np.abs(height[::-1]-1))
    i_10 = np.argmin(np.abs(height[::-1]-10))
    i_50 = np.argmin(np.abs(height[::-1]-50))

    # break up shap vals
    shap_dict = {
        "xx_yy": [], # in 2D, out 2D
        "xx_y":  [], # in 2D, out 1D
        "x_yy": [],  # in 1D, out 2D
        "x_y": [],   # in 1D, out 1D
    } 
    def get_size(var, norm_file):
        if var in norm_file.keys():
            l = len(norm_file[var]["mean"].shape)
            if l == 1:
                l = norm_file[var]["mean"].shape[0]
            else:
                l = 1
        elif var in ["h2o", "cl", "q"]:
            l=norm_file["extra_3d_ta"]["mean"].shape[0]
        elif var in ["clt", "albvisdir", "albvisdif", "albnirdir", "albnirdif", "rlds_rld"]:
            l = 1
        else:
            raise ValueError(f"I don't know the size of {var}")
        return l
    in_counter = 0
    x_var  = []
    xx_var = []
    y_var  = []
    yy_var = []
    for in_idx, in_var in enumerate(variables["in_vars"]):
        lx = get_size(in_var, norm_file)
        if lx == 1:
            x_var.append(label_translate(in_var))
        else:
            xx_var.append(label_translate(in_var))
        out_counter = 0 
        temp_x_y   = []
        temp_xx_y  = []
        temp_x_yy  = []
        temp_xx_yy = []
        for out_idx, out_var in enumerate(variables["out_vars"]):
            ly = get_size(out_var, norm_file)
            if ly==1 and in_idx==0:
                y_var.append(label_translate(out_var))
            elif ly>1 and in_idx==0:
                yy_var.append(label_translate(out_var))
            v = vals[in_counter:in_counter+lx, out_counter:out_counter+ly].squeeze()
            
            if lx == 1 and ly == 1:
                temp_x_y.append(v)  
            elif lx == 1 and ly > 1:
                temp_x_yy.append(v[::-1])
            elif lx > 1 and ly > 1:
                temp_xx_yy.append(v.T[::-1, ::-1])
            elif lx > 1 and ly == 1:
                temp_xx_y.append(v[::-1])
            out_counter += ly
        in_counter += lx
        shap_dict["x_y"].append(np.vstack(temp_x_y)) if len(temp_x_y) else []
        shap_dict["xx_y"].append(np.vstack(temp_xx_y)) if len(temp_xx_y) else []
        shap_dict["x_yy"].append(np.vstack(temp_x_yy)) if len(temp_x_yy) else []
        shap_dict["xx_yy"].append(np.vstack(temp_xx_yy)) if len(temp_xx_yy) else []
        
    for mode in ["xx_yy", "x_y", "xx_y", "x_yy", ]:
        
        arr = np.array(shap_dict[mode])
        
        vals=np.hstack(arr) if mode != "x_yy" else np.vstack(arr).T
        plt.figure(figsize=fig_specs[mode]["figsize"])
        #plt.title(mode)
        plt.pcolor(vals, cmap="Reds",
                    #marker="s", s=10,
                    norm=mpl.colors.SymLogNorm(
                        linthresh=fig_specs[mode]["vmin"], 
                        linscale=1.,
                        vmin=0, 
                        vmax=fig_specs[mode]["vmax"],
                        base=10)
                   )
        if "xx" in mode:
            n_xx = len(xx_var)
            ticks = np.concatenate([np.array([i_1,i_10,i_50])+(i*h) for i in range(n_xx)])
            for i, x in enumerate(xx_var):
                plt.text(x=i+47*i+10 ,y=fig_specs[mode]["yshift"] , s=x)
            plt.xticks(ticks, labels=[1, 10, 50]*n_xx)
            plt.grid()

            plt.vlines([i*h for i in range(n_xx)],-1, h+1, color="black", linewidths=1)
            plt.xlim(0, n_xx*h)
        else:
            n_x=len(x_var)
            plt.xticks(np.arange(0.5,len(x_var)), labels=x_var, rotation=90)
            plt.xlim(0, n_x)
        if "yy" in mode:
            n_yy = len(yy_var)
            ticks = np.concatenate([np.array([i_1,i_10,i_50])+(i*h) for i in range(n_yy)])
            plt.yticks(ticks, labels=[1, 10, 50]*n_yy)
            for i, y in enumerate(yy_var):
                plt.text(x=fig_specs[mode]["xshift"] ,y=i+47*i+10 , s=y, rotation=90)
            plt.hlines([i*h-0.5 for i in range(n_yy)],-1, h+1, color="black", linewidths=1)
            plt.ylim(0, n_yy*h)
        else:
            n_y = len(y_var)
            plt.yticks(np.arange(0.5,len(y_var)), labels=y_var)
            plt.ylim(0, n_y)
            
        plt.colorbar(label="Shapley values", extend="max")
        plt.tight_layout()
        plt.show()
        #plt.savefig(kwargs["save"])
        plt.close()
