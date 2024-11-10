from captum.attr import GradientShap
from tqdm import tqdm
import torch
import numpy as np


def calculate_mean_shapley(baseline_model, args):
    gs = GradientShap(baseline_model)
    X = []
    for i in range(args.coarse_test.__len__()):
        x, y = args.coarse_test.__getitem__(i)
        X.append(torch.from_numpy( x ))
    X = torch.cat(X)

    attr_list = []
    if "HR" in args.model_type:    
        for i in tqdm(range(47)):
            attributions, delta = gs.attribute(X[::100], X, target=(i,0), return_convergence_delta=True)
            attr_list.append(attributions.detach().numpy())
    else:
        for i in tqdm(range(len(args.variables["out_vars"]))):
            attributions, delta = gs.attribute(X[::10], X, target=(i), return_convergence_delta=True)
            attr_list.append(attributions.detach().numpy())
    attrs = np.array(attr_list)

    if "HR" in args.model_type:
        vals = np.mean(np.abs((attrs.T[:-47,:])), axis=1)
    elif args.model_type=="SW_FLUX":
        vals = np.mean(np.abs(attrs.T[:-2,:]), axis=1)
    else:
        vals = np.mean(np.abs(attrs.T[:,:]), axis=1)

    print("Shape:", vals.shape)
    
    return vals