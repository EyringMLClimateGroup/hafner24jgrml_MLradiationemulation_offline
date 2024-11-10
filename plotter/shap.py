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

def plot_interpret_values(vals, height, variables, norm_file, kwargs):
    x, y = np.mgrid[0:len(vals):1,0:len(vals[0]):1]
    
    
    mpl.rcParams['font.size'] = '36'
    
    print(np.argmin(np.abs(height-50)))

    plt.figure(figsize=(42,5))
    start_x, end_x, xticks, xlabels = get_start_end_in_list(variables["in_vars"], norm_file, height)
    print(xticks, xlabels)
    start_y, end_y, yticks, ylabels = get_start_end_in_list(variables["out_vars"],  norm_file, height)
    plt.vlines(end_x-0.5, -1, np.max(y)+1, color="black", linewidths=1)
    plt.hlines(end_y[:-1]-0.5, -1, np.max(x)+1, color="black", linewidths=1)
    plt.xlim(-0.50, end_x[-1]-0.5)
    plt.ylim(-0.5, np.max(y)+0.5)
    xcenters = (start_x+end_x)/2
    ycenters = (start_y+end_y)/2
    
    plt.pcolor(x,y,np.flip(vals),  cmap="Reds", norm=mpl.colors.SymLogNorm(linthresh=kwargs["vmin"], linscale=1.,vmin=0, base=10))
 
    prev_2d = 0
    for i, x in enumerate(xcenters):
        s = label_translate(variables["in_vars"][-i-1])
        
        if end_x[i]-start_x[i] >2:
            
            shift = len(s)
            prev_2d = 0
            rotate = 0
        else:
            if "frac" in s:
                shift = 13 +prev_2d
            elif len(s)==0:
                prev_2d -=2
                continue
            else:
                shift = 5 + prev_2d
            prev_2d -=4
            rotate = 90
            
        plt.text(x = x-shift, y= -1-kwargs["yshift"]*(1 + rotate/(90*3)),s= s, rotation = rotate ) 
    for i, y in enumerate(ycenters):
        s = label_translate(variables["out_vars"][-i-1])
        plt.text(x = kwargs["yxshift"], y= y-kwargs["yyshift"],s= s, rotation=kwargs["yrotate"] ) 
     
    plt.xticks(ticks=xticks-0.5, labels=xlabels)
    plt.yticks(ticks=yticks-0.5, labels=ylabels)

    plt.colorbar(label="Shapley values", extend="max", pad=0.01)
    if "HR" in kwargs['mtype']:
        plt.grid()
    plt.show()
    plt.close()