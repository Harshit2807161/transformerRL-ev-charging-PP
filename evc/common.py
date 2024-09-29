import os, datetime
#import numpy as np
import torch as tt
#import matplotlib.pyplot as plt
from io import BytesIO
now = datetime.datetime.now
fdate = datetime.datetime.strftime
pdate = datetime.datetime.strptime



def pjs(*paths):
    res = ''
    for p in paths: res = os.path.join(res, p)
    return res
def pj(path, sep='/'): return pjs(*path.split(sep))

def create_default_dirs():
    os.makedirs('__models__', exist_ok=True)
    os.makedirs(pjs('__models__', 'pp'), exist_ok=True)
    os.makedirs(pjs('__models__', 'rl'), exist_ok=True)

def save_state(model, path:str): 
    r""" simply save the state dictionary """
    tt.save(model.state_dict(), path) 

def load_state(model, path:str): 
    r""" simply load the state dictionary """
    model.load_state_dict(tt.load(path))

def make_clone(model, detach:bool=False, set_eval:bool=False):
    r""" Clone a model using memory buffer

    :param model:    an ``nn.Module`` to clone
    :param detach:   if True, sets the ``requires_grad`` to `False` on all of the parameters of the cloned model
    :param set_eval: if True, calls ``eval()`` on cloned model

    :returns: ``nn.Module``

    .. seealso::
        :func:`~known.ktorch.common.make_clones`
        :func:`~known.ktorch.common.clone_model`
    """
    buffer = BytesIO()
    tt.save(model, buffer)
    buffer.seek(0)
    model_copy = tt.load(buffer)
    if detach:
        for p in model_copy.parameters(): p.requires_grad=False
    if set_eval: model_copy.eval()
    buffer.close()
    del buffer
    return model_copy

def make_clones(model, n_copies:int, detach:bool=False, set_eval:bool=False):
    r""" Clone a model multiple times using memory buffer

    :param model:    an ``nn.Module`` to clone
    :param n_copies: number of copies to be made
    :param detach:   if True, sets the ``requires_grad`` to `False` on all of the parameters of the cloned model
    :param set_eval: if True, calls ``eval()`` on cloned model

    :returns: tuple of ``nn.Module``

    .. seealso::
        :func:`~known.ktorch.common.make_clone`
        :func:`~known.ktorch.common.clone_model`
    """
    buffer = BytesIO()
    
    tt.save(model, buffer)
    model_copies = []
    for _ in range(n_copies):
        buffer.seek(0)
        model_copy = tt.load(buffer)
        if detach:
            for p in model_copy.parameters(): p.requires_grad=False
        if set_eval: model_copy.eval()
        model_copies.append(model_copy)
    buffer.close()
    del buffer
    return tuple(model_copies)

def clone_model(model, n_copies:int=1, detach:bool=False, set_eval:bool=False):
    r""" Clone a model multiple times using memory buffer

    :param model:    an ``nn.Module`` to clone
    :param n_copies: number of copies to be made
    :param detach:   if True, sets the ``requires_grad`` to `False` on all of the parameters of the cloned model
    :param set_eval: if True, calls ``eval()`` on cloned model

    :returns: single ``nn.Module`` or tuple of ``nn.Module`` based on ``n_copies`` argument

    .. note:: This is similar to :func:`~known.ktorch.common.make_clone` and :func:`~known.ktorch.common.make_clones` but
        returns tuple or a single object based on `n_copies` argument

    """
    assert n_copies>0, f'no of copies must be atleast one'
    return (make_clone(model, detach, set_eval) if n_copies==1 else make_clones(model, n_copies, detach, set_eval))

class REMAP:
    def __init__(self,Input_Range, Mapped_Range) -> None:
        self.input_range(Input_Range)
        self.mapped_range(Mapped_Range)

    def input_range(self, Input_Range):
        self.Li, self.Hi = Input_Range
        self.Di = self.Hi - self.Li
    def mapped_range(self, Mapped_Range):
        self.Lm, self.Hm = Mapped_Range
        self.Dm = self.Hm - self.Lm

    def __call__(self, i): return self.in2map(i)
    
    def map2in(self, m):
        return ((m-self.Lm)*self.Di/self.Dm) + self.Li
    def in2map(self, i):
        return ((i-self.Li)*self.Dm/self.Di) + self.Lm

class dummy:
    def __init__(self, space) -> None:
        self.space = space
    def predict(self, obs, deterministic=True):
        return self.space.sample(), None

class user:
    def __init__(self, space) -> None:
        self.n = space.n
    def predict(self, obs, deterministic=True):
        return ( int(input()) % self.n ), None
    
    
