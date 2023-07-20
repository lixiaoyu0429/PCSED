import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorly as tl
from pathlib import Path
import torch
import argparse
import csv

tl.set_backend('pytorch')

matPath = Path('material')

def load_nk(csv_file,lambda_list,lambda_unit='nm'):
    '''
    读入材料nk值csv文件，再根据lambda_list插值出nk曲线
    @param csv_file: csv文件名
    @param lambda_list: 插值的lambda列表
    @param lambda_unit: csv文件中lambda的单位，'nm'或'um'，默认为'nm'
    @return: 插值后的nk曲线
    '''
    raw_wl = []
    raw_data = []
    multiplier = 1
    if lambda_unit == 'um':
        multiplier = 1000
    with open(csv_file) as csvFile:
        csvReader = csv.reader(csvFile)
        for row in csvReader:
            if row[0] == '':
                break
            raw_wl.append(float(row[0])*multiplier)
            raw_data.append(complex(float(row[1]), float(row[2])))

    
    return np.interp(lambda_list,raw_wl,raw_data)


def make_nx2x2_array(n, a, b, c, d, **kwargs):
    """
    Makes a nx2x2 tensor [[a,b],[c,d]] x n
    """

    my_array = tl.zeros((n, 2, 2), **kwargs)
    my_array[:, 0, 0] = a
    my_array[:, 0, 1] = b
    my_array[:, 1, 0] = c
    my_array[:, 1, 1] = d
    return my_array


def coh_tmm_normal_spol_spec(n_array, d_list, lam_vac_list):
    n_array = tl.tensor(n_array, dtype=torch.complex64)
    d_list = tl.tensor(d_list, dtype=float)
    lam_vac_list = tl.tensor(lam_vac_list, dtype=float)

    # Input tests

    assert d_list[0] == d_list[-1] == inf, 'd_list must start and end with inf!'

    num_layers = n_array.shape[1]
    num_lam = lam_vac_list.shape[0]


    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.
    # kz_list = 2 * np.pi * n_list/ lam_vac
    kz_array = 2 * tl.pi * n_array / lam_vac_list.reshape(-1,1)

    # delta is the total phase accrued by traveling through a given layer.
    # Ignore warning about inf multiplication
    olderr = np.seterr(invalid='ignore')
    delta = kz_array * d_list   # (num_lam, num_layers)
    np.seterr(**olderr)

    # t_list[i,j] and r_list[i,j] are transmission and reflection amplitudes,
    # respectively, coming from i, going to j. Only need to calculate this when
    # j=i+1. (2D array is overkill but helps avoid confusion.)
    # t_list = zeros((num_layers, num_layers), dtype=complex)
    # r_list = zeros((num_layers, num_layers), dtype=complex)
    t_array = tl.zeros((num_lam, num_layers), dtype=torch.complex64)    # t_list[0, 1] = t_array[λ, 0]
    r_array = tl.zeros((num_lam, num_layers), dtype=torch.complex64)    # r_list[0, 1] = r_array[λ, 0]

    for i in range(num_layers-1):
        t_array[:,i] = 2 * n_array[:,i] / (n_array[:,i] + n_array[:,i+1]) 
        r_array[:,i] = ((n_array[:,i] - n_array[:,i+1] ) / (n_array[:,i]  + n_array[:,i+1]))
    # At the interface between the (n-1)st and nth material, let v_n be the
    # amplitude of the wave on the nth side heading forwards (away from the
    # boundary), and let w_n be the amplitude on the nth side heading backwards
    # (towards the boundary). Then (v_n,w_n) = M_n (v_{n+1},w_{n+1}). M_n is
    # M_list[n]. M_0 and M_{num_layers-1} are not defined.
    # My M is a bit different than Sernelius's, but Mtilde is the same.
    # M_list = zeros((num_layers, 2, 2), dtype=complex)
    M_array = tl.zeros((num_lam, num_layers, 2, 2), dtype=torch.complex64)
    Mtilde_array = make_nx2x2_array(num_lam,1, 0, 0, 1, dtype=torch.complex64)
    for i in range(1, num_layers-1):
        # M_list[i] = (1/t_list[i,i+1]) * np.dot(
        #     make_2x2_array(exp(-1j*delta[i]), 0, 0, exp(1j*delta[i]),
        #                    dtype=complex),
        #     make_2x2_array(1, r_list[i,i+1], r_list[i,i+1], 1, dtype=complex))
        M_array[:,i,:,:] = (1/t_array[:,i].reshape(-1,1,1)) * tl.matmul(
            make_nx2x2_array(num_lam, tl.exp(-1j*delta[:,i]), 0, 0, tl.exp(1j*delta[:,i]), dtype=torch.complex64),
            make_nx2x2_array(num_lam, 1, r_array[:,i], r_array[:,i], 1, dtype=torch.complex64)
        )

        Mtilde_array = tl.matmul(Mtilde_array, M_array[:,i,:,:])
    # Mtilde = np.dot(make_2x2_array(1, r_list[0,1], r_list[0,1], 1,
    #                                dtype=complex)/t_list[0,1], Mtilde)
    Mtilde_array = tl.matmul(
        make_nx2x2_array(num_lam, 1, r_array[:,0], r_array[:,0], 1, dtype=torch.complex64)/t_array[:,0].reshape(-1,1,1),
        Mtilde_array
    )

    # Net complex transmission and reflection amplitudes
    # r = Mtilde[1,0]/Mtilde[0,0]
    # t = 1/Mtilde[0,0]

    t_list = 1/Mtilde_array[:,0,0]

    # Net transmitted and reflected power, as a proportion of the incoming light
    # power.
    T = abs(t_list**2) * n_array[:,-1] / n_array[:,0]

    return T.real

def coh_tmm_normal_spol_spec_d(n_array, d_array, lam_vac_list, device='cpu')->np.ndarray:
    n_array = tl.tensor(n_array, dtype=torch.complex64, device=device)
    d_array = tl.tensor(d_array, dtype=torch.complex64, device=device)            # (num_f, num_layers)
    lam_vac_list = tl.tensor(lam_vac_list, dtype=torch.float32, device=device)

    # Input tests

    num_layers = n_array.shape[1]
    num_lam = lam_vac_list.shape[0]
    num_f = d_array.shape[0]


    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.
    # kz_list = 2 * np.pi * n_list/ lam_vac
    kz_array = 2 * tl.pi * n_array / lam_vac_list.reshape(-1,1)

    # delta is the total phase accrued by traveling through a given layer.
    # Ignore warning about inf multiplication
    olderr = np.seterr(invalid='ignore')
    # delta = kz_array * d_list   # (num_lam, num_layers)

    delta = tl.zeros((num_f, num_lam, num_layers), dtype=torch.complex64, device=device)
    for l in range(num_layers):
        delta[:,:,l] = d_array[:,l].reshape(-1,1) @ kz_array[:,l].reshape(1,-1)

    delta = delta.reshape(num_f * num_lam, num_layers)

    np.seterr(**olderr)

    # t_list[i,j] and r_list[i,j] are transmission and reflection amplitudes,
    # respectively, coming from i, going to j. Only need to calculate this when
    # j=i+1. (2D array is overkill but helps avoid confusion.)
    # t_list = zeros((num_layers, num_layers), dtype=complex)
    # r_list = zeros((num_layers, num_layers), dtype=complex)
    t_array = tl.zeros((num_lam , num_layers), dtype=torch.complex64, device=device)    # t_list[0, 1] = t_array[λ, 0]
    r_array = tl.zeros((num_lam , num_layers), dtype=torch.complex64, device=device)    # r_list[0, 1] = r_array[λ, 0]

    for i in range(num_layers-1):
        t_array[:,i] = 2 * n_array[:,i] / (n_array[:,i] + n_array[:,i+1]) 
        r_array[:,i] = ((n_array[:,i] - n_array[:,i+1] ) / (n_array[:,i]  + n_array[:,i+1]))

    t_array = t_array.repeat(num_f,1)
    r_array = r_array.repeat(num_f,1)
    # At the interface between the (n-1)st and nth material, let v_n be the
    # amplitude of the wave on the nth side heading forwards (away from the
    # boundary), and let w_n be the amplitude on the nth side heading backwards
    # (towards the boundary). Then (v_n,w_n) = M_n (v_{n+1},w_{n+1}). M_n is
    # M_list[n]. M_0 and M_{num_layers-1} are not defined.
    # My M is a bit different than Sernelius's, but Mtilde is the same.
    # M_list = zeros((num_layers, 2, 2), dtype=complex)
    Mtilde_array = make_nx2x2_array(num_f * num_lam,1, 0, 0, 1, dtype=torch.complex64, device=device)
    for i in range(1, num_layers-1):
        # M_list[i] = (1/t_list[i,i+1]) * np.dot(
        #     make_2x2_array(exp(-1j*delta[i]), 0, 0, exp(1j*delta[i]),
        #                    dtype=complex),
        #     make_2x2_array(1, r_list[i,i+1], r_list[i,i+1], 1, dtype=complex))
        _m = (1/t_array[:,i].reshape(-1,1,1)) * tl.matmul(
            make_nx2x2_array(num_f * num_lam, tl.exp(-1j*delta[:,i]), 0, 0, tl.exp(1j*delta[:,i]), dtype=torch.complex64, device=device),
            make_nx2x2_array(num_f * num_lam, 1, r_array[:,i], r_array[:,i], 1, dtype=torch.complex64, device=device)
        )

        Mtilde_array = tl.matmul(Mtilde_array, _m)
    # Mtilde = np.dot(make_2x2_array(1, r_list[0,1], r_list[0,1], 1,
    #                                dtype=complex)/t_list[0,1], Mtilde)
    Mtilde_array = tl.matmul(
        make_nx2x2_array(num_f * num_lam, 1, r_array[:,0], r_array[:,0], 1, dtype=torch.complex64, device=device)
        /t_array[:,0].reshape(-1,1,1),
        Mtilde_array
    )

    # Net complex transmission and reflection amplitudes
    # r = Mtilde[1,0]/Mtilde[0,0]
    # t = 1/Mtilde[0,0]

    t_list = 1/Mtilde_array[:,0,0]

    # Net transmitted and reflected power, as a proportion of the incoming light
    # power.
    _n = n_array.repeat(num_f,1)
    _r = _n[:,-1] / _n[:,0]
    T = abs(t_list**2) * _r
    del _n, _r, t_list, Mtilde_array, t_array, r_array, delta, n_array, d_array, kz_array

    return T.real.reshape(num_f, num_lam).cpu().numpy()

if __name__ == '__main__':
    import scipy.io as sio
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Calculate the transmission of a thin film stack.')
    parser.add_argument('-l','--layers', type=int, help='Number of layers in the stack.', required=True)
    parser.add_argument('-g', '--groups', type=int, help='Number of groups of stacks.', required=True)
    parser.add_argument('-r','--resolution', type=float, help='Spectral Resolution (nm).', required=True)
    parser.add_argument('-b','--batchsize',type=int, help='Batch size', default=1000)
    parser.add_argument('-d','--device',type=str, help='Device to use', default='cpu')
    parser.add_argument('-t','--thickness',nargs=2, type=float, help='Thickness of each layer.', required=True)
    parser.add_argument('path',metavar='path', nargs='?', type=str)

    args = parser.parse_args()



    num_layers = args.layers
    num_groups = args.groups
    batch_size = args.batchsize

    thickness = args.thickness
    min_thickness = thickness[0]
    max_thickness = thickness[1]

    resolution = args.resolution
    lambda_list = np.arange(400,1000+resolution,resolution)

    parent = Path(args.path)
    folder = parent/f'L{num_layers}_R{resolution:.1f}_T{min_thickness:.0f}-{max_thickness:.0f}'
    folder.mkdir(parents=True)

    # 定义空气
    air = np.ones_like(lambda_list,dtype=complex)

    # 载入材料nk值
    sio2 = load_nk(matPath/'SiO2new.csv',lambda_list,'nm')
    tio2 = load_nk(matPath/'TiO2new.csv',lambda_list,'nm')

    # 载入玻璃nk值
    glass = load_nk(matPath/'bolijingyuan.csv',lambda_list,'nm')


    n_list = [air]
    for j in range(num_layers):
        if j % 2 == 0:
            n_list.append(sio2)
        else:
            n_list.append(tio2)
    n_list.append(glass)

    n_array = np.array(n_list).T

    for batch in range(num_groups//batch_size):
        d_array = np.random.rand(batch_size,num_layers+2) * (max_thickness - min_thickness) + min_thickness
        result = coh_tmm_normal_spol_spec_d(n_array, d_array,lambda_list,device=args.device)
        sio.savemat(folder/f'{batch}_{batch_size}.mat',{'T':result,'d':d_array[:,1:-1],'wl':lambda_list,'n':n_array})
        print(f'\rBatch {batch+1}/{num_groups//batch_size}',end='')








