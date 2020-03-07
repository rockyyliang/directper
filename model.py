import torch
import torch.nn as nn
import torch.nn.functional as F

'''
conditional direct perception network
parametrized by modules and internal dimensions
see __name__ __main__ for example on how to initialize
'''

'''module classes'''

class FCBlock(nn.Module):
    '''sequential model of linear -> relu -> dropout'''
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.block(x)

class EndBranch(nn.Module):
    '''conditional output'''
    '''
    no batch norm, explained here
    https://stats.stackexchange.com/questions/361700/lack-of-batch-normalization-before-last-fully-connected-layer
    '''
    def __init__(self, in_dim, out_dim, layer1=256, layer2=512):
        super().__init__()
        self.block = nn.Sequential(
            FCBlock(in_dim,layer1),
            FCBlock(layer1,layer2),
            nn.Linear(layer2,out_dim)
        )
    def forward(self,x):
        return self.block(x)

'''functions that return modules'''

def return_2_fc(in_dim=1, hidden_dim=64, out_dim=64):
    '''function that returns two layer fc block'''
    sequential = nn.Sequential(
        FCBlock(in_dim, hidden_dim),
        FCBlock(hidden_dim, out_dim),
        nn.BatchNorm1d(out_dim)
    )
    return sequential

def return_lstm(in_dim, hidden_dim, num_layers):
    '''returns lstm'''
    lstm = nn.LSTM(
        input_size=in_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True
    )
    return lstm

def return_branch(in_dim, out_dim, layer1=256, layer2=256):
    '''returns conditional branch'''
    branch = EndBranch(in_dim, out_dim, layer1, layer2)
    return branch

'''network'''

class CDPNet(nn.Module):
    '''conditional direct perception'''
    def __init__(self, funcs_dict, dims_dict, idim=(3,88,200)):
        super().__init__()
        self.idim = idim
        self.func_dict = func_dict
        self.dims_dict = dims_dict

        #backbone
        self.cnn = func_dict['cnn'](num_classes=dims_dict['cnn_out'])

        #v encoder
        self.sensory = func_dict['sensory'](
            in_dim = dims_dict['sensory_in'],
            hidden_dim = dims_dict['sensory_hidden'],
            out_dim = dims_dict['sensory_out']
        )

        #unconditional branch (lstm)
        self.uncond = func_dict['uncond'](
            in_dim=dims_dict['cnn_out'],
            hidden_dim=dims_dict['uncond_hidden'],
            num_layers=dims_dict['uncond_nlayers']
        )
        self.uncond_final = nn.Linear(dims_dict['uncond_hidden'], dims_dict['uncond_out'])

        #main lstm
        self.main_lstm = func_dict['main'](
            #input is the joint image+sensory representation
            in_dim=dims_dict['cnn_out']+dims_dict['sensory_out'],
            hidden_dim=dims_dict['main_hidden'],
            num_layers=dims_dict['main_layers']
        )

        #branches
        self.out_follow = func_dict['branch'](
            in_dim=dims_dict['main_hidden'],
            out_dim=dims_dict['branch_out'],
            layer1=dims_dict['branch_1'],
            layer2=dims_dict['branch_2']
        )
        self.out_right = func_dict['branch'](
            in_dim=dims_dict['main_hidden'],
            out_dim=dims_dict['branch_out'],
            layer1=dims_dict['branch_1'],
            layer2=dims_dict['branch_2']
        )
        self.out_left = func_dict['branch'](
            in_dim=dims_dict['main_hidden'],
            out_dim=dims_dict['branch_out'],
            layer1=dims_dict['branch_1'],
            layer2=dims_dict['branch_2']
        )
        self.out_rlc = func_dict['branch'](
            in_dim=dims_dict['main_hidden'],
            out_dim=dims_dict['branch_out'],
            layer1=dims_dict['branch_1'],
            layer2=dims_dict['branch_2']
        )
        self.out_llc = func_dict['branch'](
            in_dim=dims_dict['main_hidden'],
            out_dim=dims_dict['branch_out'],
            layer1=dims_dict['branch_1'],
            layer2=dims_dict['branch_2']
        )


    def _flatten_sequence(self,x):
        '''flatten input before feeding into non recurrent layers'''
        #only do the images and speeds
        x_flat = [None, None]
        x_flat[0] = x[0].view(-1, *self.idim)
        x_flat[1] = x[1].view(-1, 1)
        #x[2] = x[2].view(-1, 1)
        return x_flat

    def _reconstruct_sequence(self, enc_list, batch_size, seq_len):
        '''get sequence dim back before feeding into lstm'''
        enc_list[0] = enc_list[0].view(batch_size, seq_len, self.dims_dict['cnn_out'])
        enc_list[1] = enc_list[1].view(batch_size, seq_len, self.dims_dict['sensory_out'])
        return enc_list

    def forward(self, x, hidden_states=(None, None)):
        batch_size = x[1].shape[0]
        seq_len = x[1].shape[1]

        #flatten speends and images
        x_flat = self._flatten_sequence(x)

        #encoding images and sensor
        encoded_imgs = self.cnn(x_flat[0])
        encoded_vs = self.sensory(x_flat[1])

        #get sequence dim back
        encoded = [encoded_imgs, encoded_vs]
        encoded = self._reconstruct_sequence(encoded, batch_size, seq_len)

        #unconditional output
        unconditional, uncond_hidden = self.uncond(encoded[0], hidden_states[0])
        unconditional = self.uncond_final(unconditional[:,-1,:])

        #form joint representation and run through main lstm
        j = torch.cat(encoded, dim=2)
        main, main_hidden = self.main_lstm(j, hidden_states[1])
        main = main[:,-1,:]

        #branches
        follow = self.out_follow(main)
        right = self.out_right(main)
        left = self.out_left(main)
        rlc = self.out_rlc(main)
        llc = self.out_llc(main)

        branch_cat = torch.cat((follow, right, left, rlc, llc), dim=1)

        return [unconditional, branch_cat], (uncond_hidden, main_hidden)

if __name__=='__main__':
    import torchvision

    func_dict = {
        'cnn':torchvision.models.resnet18,
        'sensory':return_2_fc,
        'uncond':return_lstm,
        'main':return_lstm,
        'branch':return_branch
    }

    dims_dict = {
        'sensory_in':1,
        'sensory_hidden':64,
        'sensory_out':64,
        'cnn_out':128,
        'uncond_hidden':8,
        'uncond_nlayers':1,
        'uncond_out':3,
        'main_layers':1,
        'main_hidden':8,
        'branch_out':3,
        'branch_1':256,
        'branch_2':32
    }

    model = CDPNet(func_dict, dims_dict)

    bs = 32
    seq_len=5
    idim = (3, 88, 200)

    imgs = torch.randn(bs, seq_len, *idim)
    vs = torch.randn(bs, seq_len, 1)
    X = [imgs, vs]

    print('input shape:', X[0].shape, X[1].shape)

    pred, hidden_states = model(X, (None, None))

    print('output shape:',pred[0].shape, pred[1].shape)
    print('hidden shape:')
    for hs in hidden_states:
        for h in hs:
            print(h.shape)
