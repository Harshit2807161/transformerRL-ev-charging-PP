
import torch as tt
import torch.nn as nn
#import torch.nn.functional as ff
import math
from typing import Iterable, Callable, Dict

__all__ = ['RNN', 'ELMAN', 'GRU', 'LSTM', 'MGU', 'JANET', 'MGRU', 'XRNN', 'XARNN']

# Helpers

def no_activation(x): return x

def dense_sequential(in_dim:int, layer_dims:Iterable[int], out_dim:int, 
                    actF:Callable, actL:Callable, actFA:Dict={}, actLA:Dict={}, 
                    use_bias:bool=True, use_biasL:bool=True, dtype=None, device=None ):
    r"""
    Creats a stack of fully connected (dense) layers which is usually connected at end of other networks.
    
    :param in_dim:       in_features or input_size
    :param layer_dims:   size of hidden layers
    :param out_dim:      out_features or output_size
    :param actF:         activation function at hidden layer (like ``nn.Sigmoid``)
    :param actFA:        args while initializing actF
    :param actL:         activation function at last layer  (like ``nn.Sigmoid``)
    :param actLA:        args while initializing actL
    :param use_bias:     if True, uses bias at hidden layers
    :param use_biasL:    if True, uses bias at last layer

    :returns: An instance of ``nn.Module`` 

    .. seealso::
        :class:`~known.ktorch.common.LinearActivated`
    """
    layers = []
    
    # first layer
    if layer_dims:
        layers.append(nn.Linear(in_dim, layer_dims[0], bias=use_bias, dtype=dtype, device=device))
        if actF is not None: layers.append(actF(**actFA))
        # remaining layers
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=use_bias, dtype=dtype, device=device))
            if actF is not None: layers.append(actF(**actFA))
        # last layer
        layers.append(nn.Linear(layer_dims[-1], out_dim, bias=use_biasL, dtype=dtype, device=device))
        if actL is not None: layers.append(actL(**actLA))
    else:
        layers.append(nn.Linear(in_dim, out_dim, bias=use_bias, dtype=dtype, device=device))
        if actL is not None: layers.append(actL(**actLA))
    return nn.Sequential( *layers )


# RNN Base Class

class RNN(nn.Module):

    """ Recurrent Neural Network base class
        * additional parameters defined for 2 output -  `i2o` and `o2o`
        * can choose custom activation at each gate and each output 
        * can choose custom activation at seperatly for last layer
        * if i2o_sizes is None, no additional weights are defined for `i2o`
        * if o2o_sizes is None, no additional weights are defined for `o2o`

    :param input_size:      no of features in the input vector
    :param i2h_sizes:       no of features in the hidden vector (`i2h`)
    :param i2o_sizes:    optional, no of features in the first output vector (`i2o`)
    :param o2o_sizes:   optional, no of features in the second output vector (`o2o`)
    :param dropout:         probability of dropout, dropout is not applied at the last layer
    :param batch_first:     if True, `batch_size` is assumed as the first dimension in the input
    :param i2h_bias:       if True, uses bias at the cell level gates (`i2h`)
    :param i2o_bias:        if True, uses bias at first output (`i2o`)
    :param o2o_bias:       if True, uses bias at second output (`o2o`)
    :param i2h_activations:    activations at cell level, keep `None` to use default
    :param i2o_activation:    activations at i2o level, keep `None` to use default
    :param o2o_activation:       activations at o2o level, keep `None` to use default
    :param last_activation:        last output activations, keep `None` to use `No Activation`
    :param hypers:       hyper-parameters dictionary
    :param return_sequences:     if True, returns output from all timestep only else returns last timestep only
    :param stack_output:    Behaviour dependent on ``return_sequences`` arg. 
        * If ``return_sequences==True`` and ``stack_output==True`` then stacks outputs from each timestep along seq_dim
        * If ``return_sequences==True`` and ``stack_output==False`` then returns output of each timestep as a list
        * If ``return_sequences==False`` and ``stack_output==True`` then reshapes output to match sequence shape: (batch_size, 1, input_size) or (1, batch_size, input_size)
        * If ``return_sequences==False`` and ``stack_output==False`` then returns output of shape: (batch_size, input_size)
    
    .. note:: 
        * Do not use this class directly, it is meant to provide a base class from which other RNN modules are inherited
        * Activation arguments can be a tuple like ``(nn.Tanh, {})`` or a callable like ``torch.tanh``
        * if ``batch_first`` is True, accepts input of the form ``(batch_size, seq_len, input_size)``, otherwise ``(seq_len, batch_size, input_size)``
        * The forward method returns only the output not the hidden states. Last hidden state can be accessed using ``.Ht`` variable.

    """
    
    def __init__(self,
                input_size,      
                i2h_sizes,      
                i2o_sizes=None,  
                o2o_sizes=None,  
                dropout=0.0,        
                batch_first=False,
                i2h_bias = True, 
                i2o_bias = True,
                o2o_bias = True,
                i2h_activations=None,
                i2o_activation=None,
                o2o_activation=None,
                last_activation=None,
                hypers=None,
                return_sequences=False,
                stack_output=False, 
                dtype=None,
                device=tt.device("cpu"),
                ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.i2h_sizes = tuple(i2h_sizes)
        self.n_hidden = len(self.i2h_sizes)
        self.n_last = self.n_hidden-1
        self.return_sequences=return_sequences
        self.Ht=None
        #self.return_hidden=return_hidden
        if i2o_sizes is not None:
            self.i2o_sizes = tuple(i2o_sizes)
            self.n_output = len(self.i2o_sizes)
            assert self.n_hidden==self.n_output, f'i2h_sizes should be equal to i2o_sizes, {self.n_hidden}!={self.n_output}'


            if o2o_sizes is not None:
                self.o2o_sizes = tuple(o2o_sizes)
                self.n_output2 = len(self.o2o_sizes)
                assert self.n_hidden==self.n_output2, f'i2h_sizes should be equal to o2o_sizes, {self.n_hidden}!={self.n_output2}'
                self.output_size=self.o2o_sizes[-1]
            else:
                self.o2o_sizes = None
                self.n_output2=0
                self.output_size=self.i2o_sizes[-1]


        else:
            self.i2o_sizes = None
            self.n_output=0
            if o2o_sizes is not None:
                print(f'Setting o2o_sizes requires setting i2o_sizes first')
            self.o2o_sizes = None
            self.n_output2=0
            self.output_size=self.i2h_sizes[-1]


        self.i2h_bias=i2h_bias
        self.i2o_bias=i2o_bias
        self.o2o_bias=o2o_bias


        #self.i2h_activations = i2h_activations
        #self.i2o_activation = i2o_activation
        #self.o2o_activation = o2o_activation
        #self.last_activation = last_activation
        #self.hypers = 
        

        self.batch_first = batch_first
        self.batch_dim = (0 if batch_first else 1)
        self.seq_dim = (1 if batch_first else 0)
        #self.input_dim = 2
        self.stack_output = stack_output
        
        self.null_state = [None for _ in self.i2h_sizes]
        # set dropouts
        if hasattr(dropout, '__len__'): # different droupout at each layer
            if len(dropout) < self.n_hidden-1:
                self.dropouts = list(dropout)
                while len(self.dropouts) < self.n_hidden-1: self.dropouts.append(0.0)
            else:
                self.dropouts = list(dropout[0:self.n_hidden-1])
        else:
            self.dropouts = [ dropout for _ in range(self.n_hidden-1) ]
        self.dropouts.append(0.0) # for last layer, no dropout

        # build & initialize internal parameters
        if hypers is None: hypers = {} 
        self.parameters_module = self._build_parameters_(*self.make_parameters(i2h_activations, **hypers),  
                i2o_activation, o2o_activation, last_activation, dtype, device)
        self.reset_parameters() # reset_parameters should be the lass call before exiting __init__

    def make_parameters(self, i2h_activations, hypers):
        # return has_cell_state, i2h_names, i2h_activations, act_names, hyperparam
        raise NotImplemented
    
    def _build_parameters_(self, 
                    has_cell_state, i2h_names, i2h_activations, act_names, hypers,
                    i2o_activation, o2o_activation, last_activation, dtype, device):
        self.W_names = i2h_names
        for k,v in hypers.items(): setattr(self, k, v)
        i2o_names = None
        o2o_names = None
        self.init_states = self.init_states_has_cell_state if has_cell_state else self.init_states_no_cell_state
        if self.n_output>0: 
            self._build_activations_(last_activation, 'lastA')

            i2o_names = ('i2oL',)
            if i2o_activation is None: i2o_activation = tt.relu
            self._build_activations_(i2o_activation, 'i2oA')
            if self.n_output2>0:
                o2o_names = ('o2oL',)
                if o2o_activation is None: o2o_activation = tt.relu
                self._build_activations_(o2o_activation, 'o2oA')
                self.forward_one = self.forward_one_o2o_with_cell_state if has_cell_state else self.forward_one_o2o_no_cell_state
            else:
                self.forward_one = self.forward_one_i2o_with_cell_state if has_cell_state else self.forward_one_i2o_no_cell_state
        else:
            self.forward_one = self.forward_one_i2h_with_cell_state if has_cell_state else self.forward_one_i2h_no_cell_state

        for a,n in zip(i2h_activations, act_names): self._build_activations_(a, n)


        #i2o_names, o2o_names = self._build_meta_(has_cell_state)
        if self.n_output>0:
            if self.n_output2>0:
                names=i2h_names
                input_sizes=(self.input_size,) + self.o2o_sizes[:-1]
                n = len(i2h_names)
                weights = [[] for _ in range(n)]
                for in_features, cat_features, out_features in zip(input_sizes, self.i2h_sizes, self.i2h_sizes):
                    for j in range(n):
                        weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                self.i2h_bias, dtype=dtype, device=device))
                        
                for name,weight in zip(i2h_names, weights): setattr(self, name, nn.ModuleList(weight))
                

                names=names+i2o_names
                n = len(i2o_names)
                weights = [[] for _ in range(n)]
                
                for in_features, cat_features, out_features in zip(input_sizes, self.i2h_sizes, self.i2o_sizes):
                    for j in range(n):
                        weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                self.i2o_bias, dtype=dtype, device=device))

                for name,weight in zip(i2o_names, weights): setattr(self, name, nn.ModuleList(weight))
                
                names=names+o2o_names
                n = len(o2o_names)
                weights = [[] for _ in range(n)]
                is_last = len(input_sizes)-1
                for i,(in_features, cat_features, out_features) in enumerate(zip(self.i2h_sizes, self.i2o_sizes, self.o2o_sizes)):
                    if i==is_last:
                        for j in range(n):
                            weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                    self.o2o_bias, dtype=dtype, device=device))
                    else:
                        for j in range(n):
                            weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                    self.o2o_bias, dtype=dtype, device=device))

                for name,weight in zip(o2o_names, weights): setattr(self, name, nn.ModuleList(weight))
                
            else:
                names=i2h_names
                input_sizes=(self.input_size,) + self.i2o_sizes[:-1]
                n = len(i2h_names)
                weights = [[] for _ in range(n)]
                for in_features, cat_features, out_features in zip(input_sizes, self.i2h_sizes, self.i2h_sizes):
                    for j in range(n):
                        weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                self.i2h_bias, dtype=dtype, device=device))
                        
                for name,weight in zip(i2h_names, weights): setattr(self, name, nn.ModuleList(weight))
                
                names=names+i2o_names
                n = len(i2o_names)
                weights = [[] for _ in range(n)]
                is_last = len(input_sizes)-1
                for i,(in_features, cat_features, out_features) in enumerate(zip(input_sizes, self.i2h_sizes, self.i2o_sizes)):
                    if i==is_last:
                        for j in range(n):
                            weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                    self.i2o_bias, dtype=dtype, device=device))
                    else:
                        for j in range(n):
                            weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                    self.i2o_bias, dtype=dtype, device=device))

                for name,weight in zip(i2o_names, weights): setattr(self, name, nn.ModuleList(weight))

        else:
            names=i2h_names
            n = len(i2h_names)
            weights = [[] for _ in range(n)]
            input_sizes=(self.input_size,) + self.i2h_sizes[:-1]
            for in_features, out_features in zip(input_sizes, self.i2h_sizes):
                for j in range(n):
                    #print(f'{j=}, {in_features=}, {out_features=}')
                    weights[j].append(nn.Linear(in_features + out_features, out_features, 
                            self.i2h_bias, dtype=dtype, device=device))

            for name,weight in zip(i2h_names, weights): setattr(self, name, nn.ModuleList(weight))
        return tuple([getattr(self, name) for name in names])

    def _build_activations_(self, activation_arg, name):
        if activation_arg is None: activation_arg = no_activation
        if hasattr(activation_arg, '__len__'):
            # activation_arg is like activation_arg=(nn.Tanh, {})
            actModule = activation_arg[0]
            actArgs = activation_arg[1]
            setattr(self, name, actModule(**actArgs))
        else:
            # activation_arg is like activation_arg=tt.tanh
            setattr(self, name, activation_arg)

    def reset_parameters(self):
        for modulelist in self.parameters_module:
            for hs,m in zip(self.i2h_sizes,modulelist):
                stdv = 1.0 / math.sqrt(hs) if hs > 0 else 0
                for w in m.parameters(): nn.init.uniform_(w, -stdv, stdv)

    def init_states_no_cell_state(self, batch_size, dtype=None, device=None):
        return \
        ([tt.zeros(size=(batch_size, hs), dtype=dtype, device=device) for hs in self.i2h_sizes],)

    def init_states_has_cell_state(self, batch_size, dtype=None, device=None):
        return \
        ([tt.zeros(size=(batch_size, hs), dtype=dtype, device=device) for hs in self.i2h_sizes],
        [tt.zeros(size=(batch_size, hs), dtype=dtype, device=device) for hs in self.i2h_sizes])

    def forward_one_i2h_with_cell_state(self, x, s):
        H,C = [], []
        h,c = s
        for i in range(self.n_hidden):
            h_, c_, _ = self.i2h_logic(x, h[i], c[i], i)
            H.append(h_)
            C.append(c_)
            x = tt.dropout(h_, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,C)

    def forward_one_i2h_no_cell_state(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            h_, _ = self.i2h_logic(x, h[i], i)
            H.append(h_)
            x = tt.dropout(h_, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)
    
    def forward_one_i2o_with_cell_state(self, x, s):
        H,C = [], []
        h,c = s
        for i in range(self.n_hidden):
            h_, c_, xh = self.i2h_logic(x, h[i], c[i], i)
            H.append(h_)
            C.append(c_)
            x = self.lastA(self.i2oL[i]( xh ) ) if i==self.n_last else self.i2oA(self.i2oL[i]( xh ) )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,C)
    
    def forward_one_i2o_no_cell_state(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            h_, xh = self.i2h_logic(x, h[i], i)
            H.append(h_)
            x = self.lastA(self.i2oL[i]( xh ) ) if i==self.n_last else self.i2oA(self.i2oL[i]( xh ) )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)
        
    def forward_one_o2o_with_cell_state(self, x, s):
        H,C = [], []
        h,c = s
        for i in range(self.n_hidden):
            h_, c_, xh = self.i2h_logic(x, h[i], c[i], i)
            H.append(h_)
            C.append(c_)
            x = tt.concat( (self.i2oA(self.i2oL[i]( xh )), h_), dim=-1)
            x = self.lastA(self.o2oL[i]( x ) ) if i==self.n_last else self.o2oA(self.o2oL[i]( x ) )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,C)

    def forward_one_o2o_no_cell_state(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            h_, xh = self.i2h_logic(x, h[i], i)
            H.append(h_)
            x = tt.concat( (self.i2oA(self.i2oL[i]( xh )), h_), dim=-1)
            x = self.lastA(self.o2oL[i]( x ) ) if i==self.n_last else self.o2oA(self.o2oL[i]( x ) )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)
    
    def forward(self, Xt, Ht=None, future=0, reverse=False):
        r""" Applies forward pass through the entire input sequence 
        
        :param Xt:  input sequence
        :param H:   hidden states from previous timestep
        :param future:  Number of future timesteps to predict, works only when ``input_size == output_size``
        :param reverse: It True, processes sequence in reverse order
        """
        if Ht is None: Ht=self.init_states(Xt.shape[self.batch_dim], Xt.dtype, Xt.device)
        #Ht=[h] #<------ no need to store all the hidden states
        Yt = [] #<---- outputs at each timestep
        timesteps = reversed(tt.split(Xt, 1, dim=self.seq_dim)) \
                    if reverse else tt.split(Xt, 1, dim=self.seq_dim)
        for xt in timesteps: 
            x = xt.squeeze(dim=self.seq_dim)
            y, Ht = self.forward_one(x, Ht)
            Yt.append(y)
            #Ht.append(h_)
            #Ht=h_

        #<--- IMP: future arg will work only when (input_size == hidden_size of the last layer)
        for _ in range(future):
            x = Yt[-1]
            y, Ht = self.forward_one(x, Ht)
            Yt.append(y)
            #Ht.append(h_)
            #Ht=h_
        
        self.Ht = Ht
        if self.return_sequences:
            out= (tt.stack(Yt, dim=self.seq_dim) if self.stack_output else Yt)
        else:
            out= (y.unsqueeze(dim=self.seq_dim) if self.stack_output else y)
            
        return  out

    @tt.no_grad()
    def copy_torch(self, model):
        sd = model.state_dict()
        for i in range(self.n_hidden):
            ihW, hhW = sd[f'weight_ih_l{i}'], sd[f'weight_hh_l{i}']

            for n, ihw in enumerate(self.W_names):
                ll = getattr(self, ihw)
                ll[i].weight.copy_(
                        tt.cat((ihW[n*self.i2h_sizes[i]:(n+1)*self.i2h_sizes[i],:],
                        hhW[n*self.i2h_sizes[i]:(n+1)*self.i2h_sizes[i],:]), dim=1)
                        )
            if self.i2h_bias:
                ihB = sd[f'bias_ih_l{i}']
                hhB = sd[f'bias_hh_l{i}']

                for n, ihw in enumerate(self.W_names):
                    ll = getattr(self, ihw)
                    ll[i].bias.copy_(
                            ihB[n*self.i2h_sizes[i]:(n+1)*self.i2h_sizes[i]] + \
                            hhB[n*self.i2h_sizes[i]:(n+1)*self.i2h_sizes[i]]
                            )


# RNN Derived Class

class ELMAN(RNN):
    r"""
    Defines `Elman RNN <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>`__
    """
    
    def make_parameters(self, i2h_activations, **hypers):
        has_cell_state=False
        i2h_names =     ('ihL', ) 
        act_names =     ('actX', )
        hyperparam=dict()
        if not i2h_activations: i2h_activations = (tt.tanh, )
        assert len(i2h_activations)==1, f'need 1 activation for {__class__}'
        return has_cell_state, i2h_names, i2h_activations, act_names, hyperparam

    def i2h_logic(self, xi, hi, i ):
        xh = tt.concat( (xi, hi), dim=-1)
        h_ = self.actX(self.ihL[i]( xh ))
        return h_, xh

class GRU(RNN):
    r"""
    Defines `GRU RNN <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`__
    """
    
    def make_parameters(self, i2h_activations, **hypers):
        has_cell_state=False
        i2h_names =     ('irL', 'izL',  'inL')
        act_names =     ('actR','actZ', 'actN', )
        hyperparam=dict()
        if not i2h_activations: i2h_activations = (tt.sigmoid, tt.sigmoid, tt.tanh, )
        assert len(i2h_activations)==3, f'need 3 activation for {__class__}'
        return has_cell_state, i2h_names, i2h_activations, act_names, hyperparam

    def i2h_logic(self, xi, hi, i ):
        xh = tt.concat( (xi, hi), dim=-1)
        R = self.actR(self.irL[i]( xh ))
        Z = self.actZ(self.izL[i]( xh ))
        xr = tt.concat( (xi, R*hi), dim=-1)
        N = self.actN(self.inL[i]( xr ))
        h_ = (1-Z) * N + (Z * hi)  #x = (1-Z) * h[i] + (Z * N) 
        return h_, xh

class LSTM(RNN):
    r"""
    Defines `LSTM RNN <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>`__
    """
    
    def make_parameters(self, i2h_activations, **hypers):
        has_cell_state=True
        i2h_names =     ('iiL', 'ifL', 'igL', 'ioL')
        act_names =     ('actI','actF', 'actG', 'actO', 'actC' )
        hyperparam=dict()
        if not i2h_activations: i2h_activations = (tt.sigmoid, tt.sigmoid, tt.tanh, tt.sigmoid, tt.tanh, )
        assert len(i2h_activations)==5, f'need 5 activation for {__class__}'
        return has_cell_state, i2h_names, i2h_activations, act_names, hyperparam

    def i2h_logic(self, xi, hi, ci, i ):
        xh = tt.concat( (xi, hi), dim=-1)
        I = self.actI(self.iiL[i]( xh ))
        F = self.actF(self.ifL[i]( xh ))
        G = self.actG(self.igL[i]( xh ))
        O = self.actO(self.ioL[i]( xh ))
        c_ = F*ci + I*G
        h_ = O * self.actC(c_)
        return h_, c_, xh

class MGU(RNN):
    r"""
    Defines `MGU RNN <https://arxiv.org/pdf/1603.09420.pdf>`__
    """

    def make_parameters(self, i2h_activations, **hypers):
        has_cell_state=False
        i2h_names =     ('ifL', 'igL')
        act_names =     ('actF', 'actG')
        hyperparam=dict()
        if not i2h_activations: i2h_activations = (tt.sigmoid, tt.tanh, )
        assert len(i2h_activations)==2, f'need 2 activation for {__class__}'
        return has_cell_state, i2h_names, i2h_activations, act_names, hyperparam
    
    def i2h_logic(self, xi, hi, i ):
        xh = tt.concat( (xi, hi), dim=-1)
        F = self.actF(self.ifL[i]( xh )) #self.ifL[i]( xh - self.beta ) 
        xf = tt.concat( (xi, F*hi), dim=-1)
        G = self.actG(self.igL[i]( xf ))
        h_ = (1-F)*hi + F*G
        return h_, xh
    
class JANET(RNN):
    r"""
    Defines `JANET RNN <https://arxiv.org/pdf/1804.04849.pdf>`__
    """

    def make_parameters(self, i2h_activations, **hypers):
        has_cell_state=False
        i2h_names =     ('ifL', 'igL')
        act_names =     ('actF', 'actG')
        hyperparam=dict(beta = hypers.pop('beta', 1.0))
        self.i2h_logic = self.i2h_logic_without_beta  if hyperparam['beta'] == 0 else self.i2h_logic_with_beta
        if not i2h_activations: i2h_activations = (tt.sigmoid, tt.tanh, )
        assert len(i2h_activations)==2, f'need 2 activation for {__class__}'
        return has_cell_state, i2h_names, i2h_activations, act_names, hyperparam
        
    def i2h_logic_without_beta(self, xi, hi, i ):
        xh = tt.concat( (xi, hi), dim=-1)
        F = self.actF(self.ifL[i]( xh )) #self.ifL[i]( xh - self.beta ) 
        G = self.actG(self.igL[i]( xh ))
        h_ = F*hi + (1-F)*G
        return h_, xh
    
    def i2h_logic_with_beta(self, xi, hi, i ):
        xh = tt.concat( (xi, hi), dim=-1)
        S = self.ifL[i]( xh )
        F = self.actF( S ) 
        F_ = self.actF( S - self.beta) 
        G = self.actG(self.igL[i]( xh ))
        h_ = F*hi + (1-F_)*G
        return h_, xh

class MGRU(RNN):
    r"""
    Defines `Modified GRU RNN`
    """
    
    def make_parameters(self, i2h_activations, **hypers):
        has_cell_state=False
        i2h_names =     ('irL', 'izL',  'inL')
        act_names =     ('actR','actZ', 'actN', )
        hyperparam=dict()
        if not i2h_activations: i2h_activations = (tt.sigmoid, tt.sigmoid, tt.tanh, )
        assert len(i2h_activations)==3, f'need 3 activation for {__class__}'
        return has_cell_state, i2h_names, i2h_activations, act_names, hyperparam

    def i2h_logic(self, xi, hi, i ):
        xh = tt.concat( (xi, hi), dim=-1)
        R = self.actR(self.irL[i]( xh ))
        Z = self.actZ(self.izL[i]( xh ))
        N = self.actN(self.inL[i]( xh ))
        h_ = (Z * hi) + (R * N)
        #xr = tt.concat( (xi, R*hi), dim=-1)
        #N = self.actN(self.inL[i]( xr ))
        #h_ = (1-Z) * N + (Z * hi)  #x = (1-Z) * h[i] + (Z * N) 
        return h_, xh


# RNN Xtended Class

class XRNN(nn.Module):
    r"""
    Xtended-RNN: with optional dense connection applied only at the last timestep
    """

    def __init__(self, 
            coreF,
            bidir = False,
            fc_layers = None,
            fc_act = None,
            fc_last_act = None,
            fc_bias = True,
            **kwargs
        ) -> None:
        super().__init__()
        #self.coreF=coreF
        self.bidir=bidir
        self.coreForward = coreF(**kwargs)
        if bidir: self.coreBackward = coreF(**kwargs)

        if fc_layers is None: 
            print(f'[WARNING]:: FC not provided, will not be added.')
            # if FC is not present, then return the output as is such that it can be contateated,
            # this means we can set return_seqences == stack_output 
            if bidir:
                self.forward = \
                    self.forward_bi_pairwise_cat if self.coreForward.return_sequences and \
                    not self.coreForward.stack_output else self.forward_bi
            else:
                self.forward = self.forward_uni
        else:
            # if FC is present, then apply only at last output
            # this means we can set return_seqences to false and stack_output to false
            if self.coreForward.return_sequences:
                print(f'[WARNING]:: setting "return_sequences = False"')
                self.coreForward.return_sequences = False
                if bidir: self.coreBackward.return_sequences = False
            if self.coreForward.stack_output:
                print(f'[WARNING]:: setting "stack_output = False"')
                self.coreForward.stack_output = False
                if bidir: self.coreBackward.stack_output = False

            if fc_act is None: fc_act=(None, {})
            if fc_last_act is None: fc_last_act = (None, {})
            self.fc = dense_sequential(in_dim=(self.coreForward.output_size*2 if bidir else self.coreForward.output_size),
                    layer_dims=fc_layers, out_dim=self.coreForward.input_size, 
                    actF=fc_act[0], actL=fc_last_act[0], actFA=fc_act[1], actLA=fc_last_act[1], 
                    use_bias=fc_bias, use_biasL=fc_bias, dtype=kwargs.get('dtype',None), device=kwargs.get('device',None))
            self.forward = (self.forward_bi_FC if bidir else self.forward_uni_FC)

    def forward_uni_FC(self, X): return self.fc(self.coreForward(X))

    def forward_bi_FC(self, X):return self.fc(tt.cat(
            (self.coreForward(X, reverse=False), self.coreBackward(X, reverse=True)), dim=-1))
    
    def forward_uni(self, X): return self.coreForward(X)

    def forward_bi_pairwise_cat(self, X):
        outF = self.coreForward(X, reverse=False)
        outB = self.coreBackward(X, reverse=True)
        return [ tt.cat(( f,b ), dim=-1) for f,b in zip(outF, outB) ]

    def forward_bi(self, X):return self.fc(tt.cat(
            (self.coreForward(X, reverse=False), self.coreBackward(X, reverse=True)), dim=-1))

class XARNN(nn.Module):
    r"""
    Xtended-Attention-RNN: with optional dense connection applied at the context vector, 
    and optionally at the outputs (use fc_output=True)
    .. note::
        * Attention is applied to the outputs of cores, which depend upon (i2h, i2o, o2o)
        * if FC is present, forward methods returns one output (that of fc, taking context as input)
        * if FC is absent, forward methods returns 2-tuple - the output and context, 
            where output is a list (of outputs from all timesteps)
    """

    def __init__(self, 
            coreF,
            bidir = False,
            fc_layers = None,
            fc_act = None,
            fc_last_act = None,
            fc_bias = True,
            fc_output=True,
            **kwargs
        ) -> None:
        super().__init__()
        #self.coreF=coreF
        self.bidir=bidir
        self.coreForward = coreF(**kwargs)
        if bidir: self.coreBackward = coreF(**kwargs)
        output_size = self.coreForward.output_size*2 if bidir else self.coreForward.output_size

        if not self.coreForward.return_sequences: 
            print(f'[WARNING]:: setting "return_sequences = True"') #<---- required becuase attention is applied to all outputs
            self.coreForward.return_sequences = True
            if bidir: self.coreBackward.return_sequences = True
        if self.coreForward.stack_output:
            print(f'[WARNING]:: setting "stack_output = False"')
            self.coreForward.stack_output = False
            if bidir: self.coreBackward.stack_output = False

        if fc_layers is None: 
            print(f'[WARNING]:: FC not provided, will not be added.')
            self.fc = no_activation
            self.forward = (self.forward_bi if bidir else self.forward_uni)
        else:
            if fc_act is None: fc_act=(None, {})
            if fc_last_act is None: fc_last_act = (None, {})
            self.fc_output = fc_output
            fc_in_dim = output_size*2 if fc_output else output_size
            self.fc = dense_sequential(in_dim=fc_in_dim,
                    layer_dims=fc_layers, out_dim=self.coreForward.input_size, 
                    actF=fc_act[0], actL=fc_last_act[0], actFA=fc_act[1], actLA=fc_last_act[1], 
                    use_bias=fc_bias, use_biasL=fc_bias, dtype=kwargs.get('dtype',None), device=kwargs.get('device',None))
            
            self.forward = (self.forward_bi_FC if bidir else self.forward_uni_FC)

        self.QW = nn.Parameter(tt.rand(output_size, self.coreForward.input_size)) #nn.Linear(output_size, self.coreForward.input_size)
        self.KW = nn.Parameter(tt.rand(output_size, self.coreForward.input_size))

    def forward_attention(self, Yt, batch_size):
        Q0 = tt.matmul(Yt[-1], self.QW) # assert (Q0.shape[0]==batch_size)
        Ki = [tt.matmul(y, self.KW)  for y in Yt ] 
        Ai = tt.softmax(tt.stack([tt.stack([ tt.dot(Q0[b], k[b]) for b in range(batch_size) ]).unsqueeze(-1) for k in Ki], dim =-1), dim=-1)
        return tt.stack([ tt.stack([ y[b]*Ai[b,0,i] for i,y in enumerate(Yt) ]).sum(dim=0) for b in range(batch_size) ])

    def forward_uni(self, X):
        Yt = self.coreForward(X)
        c = self.forward_attention(Yt, X.shape[0])
        return Yt, c #<---- outputs, context

    def forward_uni_FC(self, X):
        Yt, c = self.forward_uni(X)
        return self.fc( tt.cat((Yt[-1], c), dim=-1) if self.fc_output else c)
    
    def forward_bi(self, X):
        Ytf, Ytb = self.coreForward(X, reverse=False), self.coreBackward(X, reverse=True)
        Yt = [ tt.cat((yf, yb), dim=-1) for yf, yb in zip(Ytf, Ytb) ]
        c = self.forward_attention(Yt, X.shape[0])
        return Yt, c #<---- outputs, context
    
    def forward_bi_FC(self, X):
        Yt, c = self.forward_bi(X)
        return self.fc( tt.cat((Yt[-1], c), dim=-1) if self.fc_output else c)

