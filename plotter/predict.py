import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as cl
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib as mpl
font={"size": 12}

matplotlib.rc("font", **font)

def plot_loss(history, fname, var="loss",scale="linear", title="", ylabel="Loss", folder=""):
    tl = history.history[var]
    vl = history.history[f"val_{var}"]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.plot(tl, label="Training")
    ax1.plot(vl, label="Validation")
    ax1.legend()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.set_yscale(scale)

    ax2.plot(history.history["lr"], c="green")
    ax2.set_ylabel("Learning Rate")
    ax2.tick_params(axis='y', colors='green')
    ax2.set_yscale("log")
    ax2.yaxis.label.set_color('green')

    ax1.grid(which='both')
    plt.tight_layout()
    plt.savefig(folder+fname+var+"_loss.png")
    plt.show()
    plt.close()

def eval_plots(true_toa, pred_toa, var, bins=20, folder=""):
    """
    Generate evaluation plots for true and predicted TOA (Top of Atmosphere) values.
    Parameters:
    true_toa (array-like): Array of true TOA values.
    pred_toa (array-like): Array of predicted TOA values.
    var (str): Variable name for the TOA values.
    bins (int, optional): Number of bins for the histograms. Default is 20.
    folder (str, optional): Folder path to save the generated plots. Default is an empty string.
    Returns:
    None
    """

    cmap = "viridis"
    import matplotlib.colors as cl
    norm = cl.LogNorm()
    #combined_hist2d(true_toa, pred_toa, bins=bins, savename=folder+var+"baseline_comb.png")

    y = np.subtract(true_toa, pred_toa)
    #print(y.shape)
    h, xedges, yedges = np.histogram2d(true_toa, y, bins=bins)
    #print(xedges, yedges)
    xcenters = (xedges[1:] + xedges[:-1])/2
    ycenters = (yedges[1:] + yedges[:-1])/2

    mse = np.zeros_like(xcenters)
    std_p = np.zeros_like(xcenters)
    std_m = np.zeros_like(xcenters)
    for i in range(len(xcenters)):
        try:
            mse[i] = np.average(ycenters, weights=h[i,:])
        except:
            mse[i] = 0
        diff = ycenters-mse[i]
        sp = diff>0
        sm = diff<0
        if np.sum(h[i,sp])==0:
            std_p[i] =0
        else:
            std_p[i] = np.sqrt(np.average(diff[sp]**2, weights=h[i,sp]))
        if np.sum(h[i,sm])==0:
            std_m[i] = 0
        else:
            std_m[i] = np.sqrt(np.average(diff[sm]**2, weights=h[i,sm]))


    fig = plt.figure(figsize=(6,6))
    im=plt.hist2d(true_toa, y, bins = bins, cmap = cmap, norm=norm, density=True, alpha=0.5)
    plt.xlabel(f"True {var} [$W/m^2$]")
    plt.ylabel(f"True - Predicted [$W/m^2$]")
    text = "$\\bar{x}$ = " + str(np.round(np.mean(y), 2)) + "; $\\bar{|x|}$ = " + str(np.round(np.mean(np.abs(y)), 2)) + "; median: " + str(np.round(np.median(y), 2))
    plt.title(text)
    plt.plot([np.min(true_toa), np.max(true_toa)], [0,0], color="red", linestyle="dashed", alpha=0.5)

    plt.plot(xcenters, mse, color="red")
    plt.fill_between(xcenters, mse+1.96*std_p, mse-1.96*std_m, color = "red", alpha=0.3)
    plt.subplots_adjust(right=1)
    cbar_ax = fig.add_axes([1, 0.075, 0.025, 0.9])
    cbar = fig.colorbar(im[3], cax=cbar_ax)
    cbar.set_label("Probability")
    plt.tight_layout()
    plt.savefig(folder+var+"_error.png")
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(6,6))
    plt.scatter(true_toa, y, alpha=0.1, s=0.2)
    plt.xlabel(f"True {var} [$W/m^2$]")
    plt.ylabel(f"Error [$W/m^2$]")
    text = "$\\bar{x}$ = " + str(np.round(np.mean(y), 2)) + "; $\\bar{|x|}$ = " + str(np.round(np.mean(np.abs(y)), 2)) + "; median: " + str(np.round(np.median(y), 2))
    plt.title(text)
    plt.plot([np.min(true_toa), np.max(true_toa)], [0,0], color="red", linestyle="dashed", alpha=0.5)
    plt.plot(xcenters, mse, color="red")
    plt.fill_between(xcenters, mse+1.96*std_p, mse-1.96*std_m, color = "red", alpha=0.3)
    plt.tight_layout()
    plt.savefig(folder+var+"_scatter.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.hist(true_toa,bins = bins,alpha = 0.7,density = True,label = 'True')
    plt.hist(pred_toa,bins = bins,density = True,label = 'Predicted',alpha = 0.7)
    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel(f'{var} [$W/m^2$]')
    plt.yscale('log')
    plt.savefig(folder+var+"_hist.png")
    plt.show()
    plt.close()

def plot_statistics_profile(var_dict, pres, main_var, xlabel, scale="log", xlim=(None, None), fname=""):
    """
    Plot statistics profile.
    Args:
        var_dict (dict): Dictionary containing the variables for plotting.
        pres (array-like): Pressure values.
        main_var (str): Main variable to plot.
        xlabel (str): Label for the x-axis.
        scale (str, optional): Scale for the y-axis. Defaults to "log".
        xlim (tuple, optional): Limits for the x-axis. Defaults to (None, None).
        fname (str, optional): File name for saving the plot. Defaults to "".
    Returns:
        None
    """
    import matplotlib as mpl
    mpl.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(var_dict[main_var], pres/100, c="C0")
    ax1.plot(var_dict["mean"], pres/100, c="C1")
    if main_var=="mae":
        ax1.fill_betweenx(pres/100, var_dict["abs_perc_5"], var_dict["abs_perc_95"] , alpha=0.3, color="C0")
    ax1.set_xlim(xlim)
    if scale=="linear":
        ax1.set_ylim((0, np.max(pres/100)*1.005))
    ax1.set_ylabel("Pressure [hPa]")
    
    if scale=="log":
        ax1.text(np.mean(ax1.get_xticks())/2, 5* 10**4, 'Mean Bias [K/d]', color='C1')
    else:
        ax1.text(np.mean(ax1.get_xticks())/2, 1200, 'Mean Bias [K/d]', color='C1')
    ax1.set_xlabel(xlabel)
    ax1.tick_params(axis='x', colors='C0')
    ax1.xaxis.label.set_color('C0')
    n_ticks = len(ax1.get_xticks())
    ax1.set_xticks(ax1.get_xticks())
    print(n_ticks)
    
    ax2.plot(var_dict["r2"], pres/100, c="green")
    ax2.set_xlabel("$R^2$")
    ax2.tick_params(axis='x', colors='green')
    ax2.xaxis.label.set_color('green')
    xmin=0
    xmax=1
    x_ticks=np.linspace(xmin, xmax, n_ticks)
    ax2.set_xlim((xmin,xmax))
    ax2.set_xticks([np.round(t, 2) for t in x_ticks])
    ax2.set_ylim((np.min(pres)/100, np.max(pres)/100))
    vname = xlabel.split()[0]
    text1 = f' $R^2$ TOA = {var_dict["r2"][0]:.2f} \n $R^2$ surf = {var_dict["r2"][-1]:.2f} \n $R^2$ = {var_dict["r2_mean"]:.2f} \n '
    text2 = f'{vname} TOA = {var_dict[main_var][0]:.2f} ({var_dict[main_var][0]/var_dict["true"][0]*100:.2f} %) \n {vname} surf = {var_dict[main_var][-1]:.2f} ({var_dict[main_var][-1]/var_dict["true"][-1]*100:.2f} %)\n {vname} = {np.mean(var_dict[main_var]):.2f} ({np.mean(var_dict[main_var]/np.abs(var_dict["true"]))*100:.2f} %) \n '
    text3 = f'Bias TOA= {var_dict["mean"][0]:.2f} ({var_dict["mean"][0]/var_dict["true"][0]*100:.2f} %) \n Bias surf = {var_dict["mean"][-1]:.2f} ({var_dict["mean"][-1]/var_dict["true"][-1]*100:.2f} %) \n Bias = {np.mean(var_dict["mean"]):.2f} ({np.mean(var_dict["mean"]/np.abs(var_dict["true"]))*100:.2f} %)\n'

    print( text1+text2+text3)
    
    ax1.grid(axis='both')
    plt.gca().invert_yaxis()
    plt.yscale(scale)
    plt.tight_layout()
    plt.savefig(fname+".png")
    plt.show()
    plt.close()

def plot_statistics_profile_height(var_dict, h, main_var, xlabel, scale="log", xlim=(None, None), fname=""):
    """
    Plot statistics as a function of height.
    Parameters:
    - var_dict (dict): Dictionary containing the variables to be plotted.
    - h (array-like): Array of heights.
    - main_var (str): Main variable to be plotted.
    - xlabel (str): Label for the x-axis.
    - scale (str, optional): Scale of the y-axis. Defaults to "log".
    - xlim (tuple, optional): Limits for the x-axis. Defaults to (None, None).
    - fname (str, optional): File name for saving the plot. Defaults to "".
    Returns:
    None
    """

    fig = plt.figure(figsize=(5,5))
    import matplotlib as mpl
    mpl.rcParams['font.size'] = '15'
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(var_dict[main_var], h, c="C0")
    ax1.plot(var_dict["mean"], h, c="C1")
    if main_var=="mae":
        ax1.fill_betweenx(h, var_dict["abs_perc_5"], var_dict["abs_perc_95"] , alpha=0.3, color="C0")
    ax1.set_ylim((np.min(h), np.max(h)))
    ax1.set_ylabel("Height [km]")
    if xlim[0] is None or xlim[1] is None:
        ax1.set_xlim(xlim)
        n_ticks = len(ax1.get_xticks())
        ax1.set_xticks(ax1.get_xticks())
    else:
        n_ticks = 7
        x_ticks=np.linspace(xlim[0], xlim[1], n_ticks)
        ax1.set_xlim(xlim)
        ax1.set_xticks(x_ticks)
    
    if scale=="log":
        ax1.text(np.mean(ax1.get_xticks())/2, 0.05, 'Mean Bias [K/d]', color='C1')
    else:
        ax1.text(np.mean(ax1.get_xticks())/2, -25, 'Mean Bias [K/d]', color='C1')
    ax1.set_xlabel(xlabel)
    ax1.tick_params(axis='x', colors='C0')
    ax1.xaxis.label.set_color('C0')
    ax2.plot(var_dict["r2"], h, c="green")
    ax2.set_xlabel("$R^2$")
    ax2.tick_params(axis='x', colors='green')
    print([t for t in ax2.get_xticks()])
    print([f'{t:.1f}' for t in ax2.get_xticks()])
    
    #ax2.set_xticklabels( [f'{t:.1f}' for t in ax2.get_xticks()]) 
    ax2.xaxis.label.set_color('green')
    xmin=0
    xmax=1
    x_ticks=np.linspace(xmin, xmax, n_ticks)
    ax2.set_xlim((xmin,xmax))
    ax2.set_xticks([np.round(t, 2) for t in x_ticks])
    #ax2.set_xticklabels()
    vname = xlabel.split()[0]
    text1 = f' $R^2$ TOA = {var_dict["r2"][0]:.4f} \n $R^2$ surf = {var_dict["r2"][-1]:.4f} \n $R^2$ = {var_dict["r2_mean"]:.4f} \n '
    text2 = f'{vname} TOA = {var_dict[main_var][0]:.4f} ({var_dict[main_var][0]/var_dict["true"][0]*100:.4f} %) \n {vname} surf = {var_dict[main_var][-1]:.4f} ({var_dict[main_var][-1]/var_dict["true"][-1]*100:.4f} %)\n {vname} = {np.mean(var_dict[main_var]):.4f} ({np.mean(var_dict[main_var]/np.abs(var_dict["true"]))*100:.2f} %) \n '
    text3 = f'Bias TOA= {var_dict["mean"][0]:.4f} ({var_dict["mean"][0]/var_dict["true"][0]*100:.4f} %) \n Bias surf = {var_dict["mean"][-1]:.4f} ({var_dict["mean"][-1]/var_dict["true"][-1]*100:.4f} %) \n Bias = {np.mean(var_dict["mean"]):.4f} ({np.mean(var_dict["mean"]/np.abs(var_dict["true"]))*100:.4f} %)\n'
    print(fname, "\n", text1+text2+text3)

    ax1.grid(axis='both')
    plt.yscale(scale)
    plt.tight_layout()
    plt.savefig(fname+"_height.png")
    plt.show()
    plt.close()



def plot_sample(true, pred, pres, samples, xlabel, scale="log", ylabel="Pressure [hPa]", xlim=(None, None), fname=""):
    """
    Plot samples of true and predicted values against pressure.
    Parameters:
    true (dict): Dictionary of true values for each sample.
    pred (dict): Dictionary of predicted values for each sample.
    pres (array-like): Array of pressure values.
    samples (list): List of sample numbers to plot.
    xlabel (str): Label for the x-axis.
    scale (str, optional): Scale for the y-axis. Defaults to "log".
    ylabel (str): Label for the y-axis.
    xlim (tuple, optional): Limits for the x-axis. Defaults to (None, None).
    fname (str, optional): File name for saving the plot. Defaults to "".
    Returns:
    None
    """

    for s in samples:
        fig = plt.figure()
        plt.title(f"Sample: {s}")
        plt.plot(true[s], pres, label="True")
        plt.plot(pred[s], pres, label="Pred")
        plt.xlim(xlim)
        
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if "Pressure" in ylabel:
            plt.gca().invert_yaxis()
        plt.yscale(scale)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname+f"_{s}.png")
        plt.show()
        plt.close()

def pres_vs_lat(mean_pres, pres, lat, data, label="", title="", summary_f = np.mean, norm=None, scale="linear", cmap="YlGnBu",ymode="pres", extend="neither", fname="",bins=90):
    """
    Plot a 2D color plot of a variable as a function of latitude and pressure or altitude.
    Parameters:
    - mean_pres (array-like): Mean pressure values.
    - pres (array-like): Pressure values.
    - lat (array-like): Latitude values.
    - data (array-like): Variable values.
    - label (str, optional): Label for the colorbar. Default is an empty string.
    - title (str, optional): Title for the plot. Default is an empty string.
    - summary_f (function, optional): Summary function to apply to the variable values. Default is np.mean.
    - norm (matplotlib.colors.Normalize, optional): Normalize instance used to scale the colorbar. Default is None.
    - scale (str, optional): Scale of the y-axis. Default is "linear".
    - cmap (str, optional): Colormap for the color plot. Default is "YlGnBu".
    - ymode (str, optional): Mode for the y-axis. Can be "pres" (pressure) or "height" (altitude). Default is "pres".
    - extend (str, optional): Extend of the colorbar. Default is "neither".
    - fname (str, optional): File name to save the plot. Default is an empty string.
    - bins (int, optional): Number of bins for the latitude histogram. Default is 90.
    Returns:
    None
    """
    
    fig = plt.figure(figsize=(7,5))
    mpl.rcParams['font.size'] = '18'
    lat *=180/np.pi
    _, edges = np.histogram(lat, bins=bins)
    x = np.zeros((bins,47))
    h = np.zeros((bins,47))
    for i in range(bins):
        idx1 = np.argwhere(edges[i]<lat)
        idx2 = np.argwhere(lat<=edges[i+1])
        idx = np.intersect1d(idx1, idx2)
        bin_data = data[idx]
        
        h[i,:] = np.mean(pres[:,idx], axis=1)
        x[i,:] = summary_f(bin_data, axis=0)
    
    y = (edges[1:]+edges[:-1])/2 
    
    p = np.array([mean_pres]*len(y))
    plt.xlabel("Latitude")

    if ymode=="pres":  
        plt.pcolor([y]*len(mean_pres), h.T/100, x.T, cmap=cmap,norm=norm)  
        plt.ylabel("Pressure [hPa]")
        plt.gca().invert_yaxis()
    elif ymode=="height":
        plt.pcolor([y]*len(mean_pres), h.T, x.T, cmap=cmap,norm=norm)
        plt.plot(y, h[:,-1]- (((h[:,-2])-(h[:,-1]))/2), c="black", linestyle="dashed", label="Avg. surf. alt.")
        #if "SW" in fname:
        plt.legend(loc=[0.8, -0.25])
        plt.ylabel("Altitude [km]")
        plt.ylim((0.1, 80))
    plt.yscale(scale)
    plt.title(title)
    plt.xlabel("Latitude [°]")
    plt.xticks(np.linspace(-90,90,7))
    
    
    plt.colorbar(label=label, extend=extend)
    plt.grid()
    plt.tight_layout()
    plt.savefig(fname+".png")
    plt.show()
    plt.close()

def pres_vs_lat_mean(mean_pres, lat, data, label="", summary_f = np.mean, norm=None, scale="linear", cmap="YlGnBu",ymode="pres", extend="neither", fname="",bins=90):
    """
    Plot the mean of a given data variable as a function of latitude.
    Parameters:
    - mean_pres (float): The mean pressure value.
    - lat (ndarray): The latitude values.
    - data (ndarray): The data variable to be plotted.
    - label (str, optional): The label for the plot. Default is an empty string.
    - summary_f (function, optional): The summary function to be applied to the data. Default is np.mean.
    - norm (Normalize, optional): The normalization object to be used for the color mapping. Default is None.
    - scale (str, optional): The scale of the y-axis. Default is "linear".
    - cmap (str, optional): The colormap to be used for the plot. Default is "YlGnBu".
    - ymode (str, optional): The mode for the y-axis. Can be "pres" (pressure) or "height". Default is "pres".
    - extend (str, optional): The extension of the colorbar. Can be "neither", "both", "min", or "max". Default is "neither".
    - fname (str, optional): The filename for saving the plot. Default is an empty string.
    - bins (int, optional): The number of bins for the latitude histogram. Default is 90.
    Returns:
    None
    """
    
    lat = np.sin(lat)
    _, edges = np.histogram(lat, bins=bins)
    x = np.zeros((bins,47))
    for i in range(bins):
        idx1 = np.argwhere(edges[i]<lat)
        idx2 = np.argwhere(lat<=edges[i+1])
        idx = np.intersect1d(idx1, idx2)
        bin_data = data[idx]
        x[i,:] = summary_f(bin_data, axis=0)
    
    y = (edges[1:]+edges[:-1])/2 
    
    p = np.array([mean_pres]*len(y))
    plt.xlabel("Latitude")

    if ymode=="pres":  
        plt.pcolor([y]*len(mean_pres), p.T/100, x.T, cmap=cmap,norm=norm)  
        plt.ylabel("Pressure [hPa]")
        plt.gca().invert_yaxis()
    elif ymode=="height":
        plt.pcolor([y]*len(mean_pres), p.T, x.T, cmap=cmap,norm=norm)
        plt.ylabel("Height [km]")
    plt.yscale(scale)
    plt.title(label)
    plt.xlabel("sin( Latitude )")
    
    
    plt.colorbar(label=label, extend=extend)
    plt.grid()
    plt.savefig(fname+".png")
    plt.show()
    plt.close()