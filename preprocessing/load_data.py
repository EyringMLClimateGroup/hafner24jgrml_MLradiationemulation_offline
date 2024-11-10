import xarray as xr
import numpy as np
from datetime import datetime
from pymeeus.Epoch import Epoch
from pymeeus.Earth import Earth


class MyMultiFileGenerator():
    """
    A generator class for loading data from multiple files.
    Args:
        batch_size (int): The batch size for each iteration.
        times (int): The number of time steps.
        cell_filter (list or None): A list of cell filters or None if no filter is applied.
        norm_dict (dict): A dictionary containing normalization parameters.
        shuffle (bool): Whether to shuffle the data at the end of each epoch.
        ncells (int): The number of cells.
    Attributes:
        rng (numpy.random.Generator): The random number generator.
        batch_size (int): The batch size for each iteration.
        times (int): The number of time steps.
        cells (list): A list of cell filters.
        len_cells (list): A list containing the length of each cell filter.
        len_set (int): The length of the dataset.
        norm_dict (dict): A dictionary containing normalization parameters.
        shuffle (bool): Whether to shuffle the data at the end of each epoch.
    Methods:
        __len__(): Returns the length of the dataset.
        idx_to_tc(idx): Converts an index to a time step and cell index.
        on_epoch_end(): Shuffles the data at the end of each epoch.
        __getitem__(idx): Retrieves an item from the dataset.
    """

    def __init__(self, batch_size, times, cell_filter, norm_dict, shuffle, ncells):
        self.rng = np.random.default_rng(3141)
        self.batch_size = batch_size
        self.times = times

        if cell_filter:
            self.cells = cell_filter
        else:
            cells = np.arange(ncells).astype('int')
            self.cells = [cells]*self.times

        self.len_cells = [len(c) for c in self.cells]
        if self.batch_size is None:
            self.len_set = times
        else:
            self.len_set =  sum([len(c)//self.batch_size for c in self.cells])
        self.norm_dict = norm_dict
        self.shuffle = shuffle

    def __len__(self):
        return int(self.len_set)

    def idx_to_tc(self, idx):
        t = 0
        while t<self.times: 
            a = sum([self.len_cells[i]//self.batch_size for i in range(t)]) # all samples used until timestep t
            if idx-a==0: # check if the batch would be in time step t
                t=t 
                break
            elif idx-a<0: # check if batch would be in previous timestep
                t=t-1
                break
            else: 
                t=t+1
        else:
            t = self.times - 1

        samples = sum([self.len_cells[i]//self.batch_size for i in range(t)]) # samples in previous files
        c = (idx - samples)*self.batch_size
        c = np.maximum(0, c)

        return t, c

    def on_epoch_end(self):
         if self.shuffle: # shuffle (elements in file) at end of epoch
            for i in range(len(self.cells)):
                self.rng.shuffle(self.cells[i])

    def __getitem__(self, idx):
        pass

class CoarseDataGenerator(MyMultiFileGenerator):
    """
    A data generator class for loading and preparing coarse data.
    Args:
        file_list (list): List of file paths.
        grid_file (str): Path to the grid file.
        bc_solar (str): Path to the solar dataset.
        vgrid (str): Path to the vertical grid dataset.
        variables (dict): Dictionary of input and output variables.
        var_to_norm (list): List of variables to normalize.
        norm_dict (dict): Dictionary containing normalization information for each variable.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        x_norm (bool, optional): Whether to normalize the input data. Defaults to False.
        batch_size (int, optional): Batch size. Defaults to None.
        x_stack (str, optional): Stack mode for input data. Defaults to "vertical".
        cell_filter (list, optional): List of cell indices to filter. Defaults to None.
        y_stack (str, optional): Stack mode for output data. Defaults to "vertical".
        **kwargs: Additional keyword arguments.
    Attributes:
        data (xarray.Dataset): Dataset containing the data.
        grid (xarray.Dataset): Dataset containing the grid information.
        bc_solar (xarray.Dataset): Dataset containing the solar data.
        vgrid (xarray.Dataset): Dataset containing the vertical grid information.
        variables (dict): Dictionary of input and output variables.
        var_to_norm (list): List of variables to normalize.
        x_norm (bool): Whether to normalize the input data.
        x_stack (str): Stack mode for input data.
        y_stack (str): Stack mode for output data.
        extra_shape (int): Shape of the extra data.
        cache (dict): Cache dictionary for storing preprocessed data.
    Methods:
        prepare(batch, varlist, stack, t=None, batch_idx=None, norm=True):
            Preprocesses the batch data.
        get_q(batch):
            Calculates the normalized q data.
        __mystack__(x1, x2, stack=None):
            Stacks the input data.
        __getitem__(idx):
            Retrieves the preprocessed data for a given index.
    """
    def __init__(self, file_list, grid_file, bc_solar, vgrid, variables, var_to_norm, norm_dict, shuffle=False, x_norm=False, batch_size=None, x_stack="vertical", cell_filter=None, y_stack="vertical", **kwargs):
        self.data = xr.open_mfdataset(file_list, decode_times=False)
        self.grid = xr.load_dataset(grid_file, decode_times=False)
        self.bc_solar = xr.load_dataset(bc_solar,  decode_times=False)
        self.vgrid = xr.load_dataset(vgrid,  decode_times=False)
        self.variables = variables
        self.var_to_norm = var_to_norm
        times = len(self.data.time)
        self.x_norm = x_norm
        self.x_stack = x_stack
        self.y_stack = y_stack
        self.extra_shape = 0
        self.cache = dict()
        if "ncells" in self.data.sizes.keys() :
            ncells = len(self.data.ncells)
            self.cell = "ncells"
        elif "cell" in self.data.sizes.keys():
            ncells = len(self.data.cell)
            self.cell = "cell"
        else:
            raise NotImplementedError("Don't know this type of cell access")
        l = [] # get max length for horizontal mode
        for k,v in norm_dict.items():
            try:
                l_temp=len(v["mean"])
            except:
                l_temp=1
            l.append(l_temp)
        self.l = np.max(l)-1
        super().__init__(batch_size, times, cell_filter, norm_dict, shuffle, ncells) 

    def prepare(self, batch, varlist, stack, t=None, batch_idx=None):
        out_arr = []
        for i, v in enumerate(varlist):
            if v=="toa":
                d =  batch["rsd"][0].values[:,np.newaxis]
            elif v=="toa_hr":
                cur_time = self.data.time[t].values
                date = datetime.strptime(str(np.round(cur_time, 3)), "%Y%m%d.%f")
                tsi = self.bc_solar.TSI[(self.bc_solar.year==date.year)& (self.bc_solar.month==date.month)].values
                this_date = Epoch(date)
                _,_,dist = Earth().geometric_heliocentric_position_j2000(this_date) 
                
                d = batch["cosmu0"].values[:,np.newaxis]*tsi/dist**2
            elif v=="h2o":
                d = batch["extra_3d_hus"]+batch["extra_3d_clw"]+batch["extra_3d_cli"]
                d = d.squeeze().values.T
            elif v=="dz":
                d=self.vgrid["dzghalf"].isel({self.cell: batch_idx})
                d = d.T
            else:
                s = batch[v].squeeze().shape
                if len(s)==2:
                    d = batch[v].squeeze().T.values
                else:
                    d = batch[v].values[:,np.newaxis]
            if self.batch_size==1 and d is not None:
                if d.shape==(1,1):
                    d = d[0]
                else:
                    d = d.squeeze()             
            if stack=="horizontal":
                d = pad_arr(d, self.l)
                
            out_arr = self.__mystack__(out_arr, d, stack)
        return out_arr

    def get_q(self, batch):
        q = batch["extra_3d_clw"]+batch["extra_3d_cli"]+ np.abs( batch["extra_3d_hus"] - np.mean(batch["extra_3d_hus"], axis=0))
        normed_q = np.where(np.sum(q, axis=0)>0, q/np.sum(q, axis=0), np.zeros_like(q))
        if self.batch_size==1:
            normed_q = normed_q.squeeze()
        return normed_q.T
    
    def __mystack__(self, x1, x2, stack=None):
        if len(x1)==0:
            if self.batch_size == 1:
                return x2[:,np.newaxis]
            return x2
        if self.batch_size == 1:
            if stack=="horizontal":
                out = np.hstack((x1, x2[:,np.newaxis]))
            elif stack=="vertical":
                out = np.vstack((x1, x2)) 
            else:
                raise NotImplementedError()
        else:   
            if stack=="horizontal":
                out = np.dstack((x1, x2))
            elif stack=="vertical":
                out = np.hstack((x1, x2))
            else:
                raise NotImplementedError()
        return out

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]    
        else:                
            if self.batch_size is None:
                t = idx
                batch_idx = self.cells[idx]
            else:
                t, c = self.idx_to_tc(idx)
                batch_idx = self.cells[t][c:c+self.batch_size]

            batch = self.data.isel({"time":t, self.cell: batch_idx})
           
            x = self.prepare(batch, self.variables["in_vars"], self.x_stack, t=t, batch_idx=batch_idx) # norm and stack
            
            y = self.prepare(batch, self.variables["out_vars"], self.y_stack)

            extra = []
            for d in self.variables["extra_in"]:
                if d in self.data.keys():
                    extra_d = self.prepare(batch, [d], self.x_stack, t=t)
                    x = self.__mystack__(x, extra_d, self.x_stack)
                    y = self.__mystack__(y, extra_d, self.y_stack)
                    extra = self.__mystack__(extra, extra_d, self.x_stack)
                else:
                    continue

            if "q" in self.variables["extra_in"]:
                q = self.get_q(batch)
                x = self.__mystack__(x, q, self.x_stack)
                y = self.__mystack__(y, q, self.y_stack)
                extra = self.__mystack__(extra, q, self.x_stack)
            
            if len(self.variables["extra_in"])>0:
                if self.batch_size==1:
                    self.extra_shape = np.array(extra).shape[0]
                else:
                    self.extra_shape = np.array(extra).shape[-1]
            #x,y = torch.from_numpy(x).float(), torch.from_numpy(y).float() 
            self.cache[idx] = x, y   
            return x, y

def pad_arr(var_to_pad, out_len):
    """
    one sided padding

    arr: array to pad of shape (n, var)
    out_len: length of output array (n, len)
    
    return:
        padded_var: padded array
    """
    s = len(var_to_pad.shape)
    if s==1:
        l_var = len(var_to_pad)
        diff = out_len - l_var
        if diff==0:
            return var_to_pad
        pad_val = var_to_pad if l_var == 1 else var_to_pad[0]
        pad = np.broadcast_to(pad_val, (diff))
        padded_var = np.hstack((pad, var_to_pad))
    else:
        l_var = len(var_to_pad[0])
        diff = out_len - l_var
        pad_val = var_to_pad if l_var == 1 else var_to_pad[0,0]
        
        n = len(var_to_pad)
        pad = np.broadcast_to(pad_val, (n, diff))
        padded_var = np.hstack((pad, var_to_pad))

    return padded_var
