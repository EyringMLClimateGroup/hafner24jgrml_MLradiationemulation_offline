import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def plot_map(lat, lon, data, name="", label="", marker_size=0.1):
    """
    Plot a map with scatter plot of data points.
    Parameters:
    - lat: array-like, latitude values in radians
    - lon: array-like, longitude values in radians
    - data: array-like, data values corresponding to each data point
    - name: str, optional, name of the plot
    - label: str, optional, label for the colorbar
    - marker_size: float, optional, size of the markers in the scatter plot
    Returns:
    None
    """
    
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.set_global()
    ax.set_title(name)
    im = ax.scatter(lon*180/np.pi, lat*180/np.pi, s=marker_size, 
            c=data, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.colorbar(im, orientation="horizontal", label=label)
    plt.show()
    plt.close()

def plot_map_lat_profile(lat, lon, data, name="", label="", marker_size=0.1, norm=None, 
                         extend="neither", fname="map", cmap="viridis", lb=False,
                        c_ticks_default=False, plot_lat_summary=True):
    """
    Plot a map with latitude profile.
    Parameters:
    - lat (array-like): Array of latitudes in radians.
    - lon (array-like): Array of longitudes in radians.
    - data (array-like): Array of data values.
    - name (str, optional): Title of the plot. Default is an empty string.
    - label (str, optional): Label for the colorbar. Default is an empty string.
    - marker_size (float, optional): Size of the markers. Default is 0.1.
    - norm (matplotlib.colors.Normalize, optional): Normalization instance used for scaling the data values. Default is None.
    - extend (str, optional): Colorbar extension. Default is "neither".
    - fname (str, optional): File name for saving the plot. Default is "map".
    - cmap (str, optional): Colormap for the scatter plot. Default is "viridis".
    - lb (bool, optional): Flag indicating whether to show coastlines in black. Default is False. For "seismic", "bwr" coastlines are black.
    - c_ticks_default (bool, optional): Flag indicating whether to use default colorbar ticks. Default is False.
    - plot_lat_summary (bool, optional): Flag indicating whether to plot the latitude profile. Default is True.
    Returns:
    None
    """
    fig = plt.figure(figsize=(6, 3))
    import matplotlib as mpl
    mpl.rcParams['font.size'] = '15'
    #ax = axes["left"]
    w=0.8
    ax = fig.add_axes([0,0,w,1], projection=ccrs.Mollweide())
    ax.set_global()
    ax.set_title(name)
    im = ax.scatter(lon*180/np.pi, lat*180/np.pi, s=marker_size, 
            c=data, transform=ccrs.PlateCarree(), norm=norm, cmap=cmap)
    if cmap in ["seismic", "bwr"] or lb==True:
        ax.coastlines(color="black")
    else:
        ax.coastlines(color="w")
    cbar = plt.colorbar( im, orientation="horizontal", label=label, extend=extend)
    if c_ticks_default == False:
        try: 
            n = cmap.N
            span = norm._vmax - norm._vmin
            if n<10:
                n_ticks = n + 1
                f=1
            else:
                n_ticks = int(n//2+1)
                f=2
            ticks = [norm._vmin + i * f *span/n for i in range(n_ticks)]
            print(ticks)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([int(t) if t.is_integer() else f"{t:.02f}" for t in ticks])
        except:
            print("use default ticks")
    else:
        print("use default ticks")

    if plot_lat_summary:
        #ax2 = axes["right"]
        ax2 = fig.add_axes([w+0.15,0.3,0.2,0.7])
        #ax2.margins(5)
        bins=90
        _, edges = np.histogram(lat, bins=bins)
        x = np.zeros(bins)
        for i in range(bins):
            idx1 = np.argwhere(edges[i]<lat)
            idx2 = np.argwhere(lat<edges[i+1])
            idx = np.intersect1d(idx1, idx2)
            bin_data = data[idx]
            x[i] = np.mean(bin_data)
        
        y = (edges[1:]+edges[:-1])/2
        
        ax2.plot(x,y*180/np.pi)
        ax2.set_ylabel("Latitude")
        ax2.set_xlabel(label)
        plt.grid()
    
    plt.tight_layout()
    plt.savefig(fname+".png", bbox_inches="tight", pad_inches=0.5)
    plt.show()
    plt.close()