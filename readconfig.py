import sys
from configparser import ConfigParser

import torchvision

'''function alias for imported models'''
resnet18 = torchvision.models.resnet18
from model import return_2_fc, return_lstm, return_branch

def readconfig(cpath):
    parser = ConfigParser()
    parser.read(cpath)

    modules_list = parser.options('modules')
    dims_list = parser.options('dims')
    scales_list = parser.options('scales')
    params_list = parser.options('params')

    #populate dictionaries for model and training params
    func_dict = {}
    g = globals()
    for m_key in modules_list:
        func_key = parser.get('modules', m_key)
        func_dict[m_key] = g[func_key]

    dims_dict = {}
    for d_key in dims_list:
        dim = parser.get('dims', d_key)
        dims_dict[d_key] = int(dim)

    scales_dict = {}
    for s_key in scales_list:
        scale = parser.get('scales', s_key)
        scales_dict[s_key] = float(scale)

    params_dict = {}
    for p_key in params_list:
        param = parser.get('params', p_key)
        try:
            params_dict[p_key] = int(param)
        except ValueError:
            params_dict[p_key] = float(param)

    return func_dict, dims_dict, scales_dict, params_dict

if __name__=='__main__':
    try:
        cpath = sys.argv[1]
    except IndexError:
        cpath = 'configs/default.ini'

    dicts = readconfig(cpath)

    print(dicts)
