import numpy as np
import copy 
import torch
from df.modules import get_device
def mcra2_estimation(ns_ps, parameters):
    n = parameters['n']
    len_val = parameters['len']
    ad = parameters['ad']
    as_val = parameters['as']
    ap = parameters['ap']
    beta = parameters['beta']
    gamma = parameters['gamma']
    alpha = parameters['alpha']
    pk = parameters['pk']
    delta = parameters['delta']

    noise_ps = parameters['noise_ps']
    pxk = parameters['pxk']
    pnk = parameters['pnk']
    pxk_old = parameters['pxk_old']
    pnk_old = parameters['pnk_old']

    pxk = alpha * pxk_old + (1 - alpha) * ns_ps
    # pnk = pxk.copy()
    pnk = copy.deepcopy(pxk)
    # pnk_old_idx = pnk_old < pxk
    pnk_old_idx = torch.where(pnk_old<pxk)
    pnk[pnk_old_idx] = gamma * pnk_old[pnk_old_idx] + (((1 - gamma) / (1 - beta)) *(pxk[pnk_old_idx] - beta * pxk_old[pnk_old_idx]))
    # pxk_old = pxk.copy()
    pxk_old = copy.deepcopy(pxk)
    # pnk_old = pnk.copy()

    pnk_old = copy.deepcopy(pnk)
    Srk = torch.zeros((len_val,1)).to(device=get_device())
    # print(Srk.shape)  (512,1)
    Srk = pxk / pnk
    # print(Srk.shape)  (512,)
    Srk_data = torch.zeros((len_val, n)).to(device=get_device())
    Srk_data[:, n - 1] = Srk
    Ikl = torch.zeros((len_val,1)).to(device=get_device())
    ikl_indx = torch.where(Srk > delta)
    # print("len printer---->",(ikl_indx))(0 - 511)
    Ikl[ikl_indx] = 1
    # print("Ikl shape ----->",Ikl.shape)(512,1)
    # pk = np.reshape(pk,(pk.shape[0],1))
    # print("pk before shape ----->",pk.shape)
    # print("*ikl shape ----->",((1-ap)*Ikl).shape)
    pk = ap * pk + (1 - ap) * Ikl
    # print("pk shape----->",pk.shape)
    adk = ad + (1 - ad) * pk
    noise_ps = adk * noise_ps + (1 - adk) * pxk

    parameters['n'] = n + 1
    parameters['pk'] = pk
    parameters['noise_ps'] = noise_ps
    parameters['pnk'] = pnk
    parameters['pnk_old'] = pnk_old
    parameters['pxk'] = pxk
    parameters['pxk_old'] = pxk_old

    return parameters
