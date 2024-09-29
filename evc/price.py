import numpy as np
import matplotlib.pyplot as plt
import datetime
import torch as tt
from torch import Tensor
#import torch.nn as nn
#import torch.optim as oo
from torch.utils.data import Dataset, DataLoader
import pandas as pd
now = datetime.datetime.now
from typing import Any, Union, Iterable, Callable, Dict, Tuple, List

class PriceData:

    def __init__(self, csv, reverse=False) -> None:
        self.csv = csv
        self.reverse = reverse
        self.prepare()

    def prepare(self):
            df = pd.read_csv(self.csv)  # Read data from the CSV file
            if self.reverse:
                df = df[::-1]  # Reverse the DataFrame

            # Extract the price data
            self.signal = df['PRICE'].to_numpy()

            # Calculate the length and number of days
            self.length = len(self.signal)
            self.n_days = int(self.length / 24)  # Assuming 24 data points per day

            # Assertion checks
            time_steps = df['TIME']
            hour = np.array([datetime.datetime.strptime(str(d), "%d-%m-%Y %H:%M").hour for d in time_steps])

            assert hour[0] == 0  # Check the first hour
            assert hour[-1] == 23  # Check the last hour



    
    
    @staticmethod
    def split_csv(csv, splits, file_names, reverse=False):
        df = pd.read_csv(csv, parse_dates=[0]) # read from csv # NOTE: reverse the sequence
        signal = df['PRICE'].to_numpy()
        times = df['TIME']
        if reverse:
            signal=signal[::-1]   
            times=times[::-1]


        length = len(signal)
        n_days = int(length/24) # how many days?
        assert(len(splits)==len(file_names))
        # assertion check
        hour = np.array([datetime.datetime.strptime(str(d), "%d-%m-%Y %H:%M").hour for d in times])

        assert(hour[0] == 0)
        assert(hour[-1] == 23)
        idays = np.where(hour==0)[0]
        splits = (np.array(splits)*n_days).astype(np.int)
        splits_ratio = []
        s=0
        for ratio,file_name in zip(splits,file_names):
            e=ratio+s
            signal_slice = signal[idays[s]:idays[e]]
            time_slice = times[idays[s]:idays[e]]
            splits_ratio.append((s,e))
            s=e
            if reverse:
                time_slice=time_slice[::-1]
                signal_slice=signal_slice[::-1]

            dd = {'TIME': time_slice, 
                  'PRICE': signal_slice,
                 }
            df = pd.DataFrame(dd)
            if (file_name is not None):
                if len(file_name)>0: df.to_csv(file_name)
        return splits_ratio
             
    def generate_normalized(csv, plot=False, file_name=None, plot_lim = 400, verbose=0):
        
        df = pd.read_csv(csv, parse_dates=[0]) # read from csv # NOTE: reverse the sequence
        signal = df['PRICE'].to_numpy()
        times = df['TIME']
        scalar = np.maximum(np.abs(np.max(signal)), np.abs(np.min(signal)))
        if scalar!=0: signal/=scalar
        df['PRICE']=signal
        
        if (file_name is not None):
            if len(file_name)>0: df.to_csv(file_name)
        
        if plot:
            
            if len(df)<=plot_lim:
                sizex, plotx, ploty = len(df), df['TIME'], df['PRICE']
            else:
                print(f'Sequence is large, Plotting only [{plot_lim}] steps')
                sizex, plotx, ploty = plot_lim, df['TIME'][0:plot_lim], df['PRICE'][0:plot_lim]
            
            plt.figure(figsize=(sizex*0.2, 6), constrained_layout=True)
            plt.plot(plotx, ploty, marker='.', linewidth=0.5, linestyle='dotted',color='black')
            #plt.show()
            plt.close()
        if verbose>0:
            print(df.info())
            if verbose>1:
                print(df.describe())
                if verbose>2:
                    print(df)
        return df 
               
    def generate(start_date_str, end_date_str, delta_seconds=3600, low=1, high=10,  normalize=False, seed=None, plot=False, file_name=None, plot_lim = 400, verbose=0): 
        if low==high:
            genF = lambda prng, size : np.zeros(size)+low
        else:
            genF = lambda prng, size : prng.uniform(low, high, size) 
        return __class__.generateF(genF, start_date_str, end_date_str, delta_seconds, normalize, seed, plot, file_name, plot_lim, verbose)
    
    def generateF(
        genF,
        start_date_str, 
        end_date_str, 
        delta_seconds=3600, 
        normalize=False, reverse=False,
        seed=None, plot=False,
        file_name=None, plot_lim = 400, verbose=0): 
        """ use format day-month-year genF like: lambda prng, size: ndarray """
        start_date = datetime.datetime.strptime( start_date_str, '%d-%m-%Y' )
        end_date = datetime.datetime.strptime( end_date_str, '%d-%m-%Y' )
        delta = datetime.timedelta(seconds=delta_seconds)
        # delta.days, delta.microseconds, delta.seconds, delta.total_seconds()
        idate = start_date
        dates = []
        while (idate<end_date):
            dates.append(idate)
            idate = idate + delta

        print('generate dates: ', len(dates))
        #for i,d in enumerate(dates):
        #    print(i, datetime.datetime.strftime(d, '%d-%m-%Y %H:%M'))

        rng = np.random.default_rng(seed)
        
        generated_dates = np.array(dates)
        generated_price = genF(rng, len(dates))
        if normalize:
            scalar = np.maximum(np.abs(np.max(generated_price)), np.abs(np.min(generated_price)))
            if scalar!=0: generated_price/=scalar
        
        if reverse:
            generated_dates=generated_dates[::-1]
            generated_price=generated_price[::-1]

        dd = {'TIME': generated_dates, 
              'PRICE': generated_price,
             }
        df = pd.DataFrame(dd)
        if (file_name is not None):
            if len(file_name)>0: df.to_csv(file_name)
        
        if plot:
            
            if len(df)<=plot_lim:
                sizex, plotx, ploty = len(df), df['TIME'], df['PRICE']
            else:
                print(f'Sequence is large, Plotting only [{plot_lim}] steps')
                sizex, plotx, ploty = plot_lim, df['TIME'][0:plot_lim], df['PRICE'][0:plot_lim]
            
            plt.figure(figsize=(sizex*0.2, 6), constrained_layout=True)
            plt.plot(plotx, ploty, marker='.', linewidth=0.5, linestyle='dotted',color='black')
            #plt.show()
            plt.close()
        if verbose>0:
            print(df.info())
            if verbose>1:
                print(df.describe())
                if verbose>2:
                    print(df)
        return df 

class SeqDataset(Dataset):
    r"""
    Sequential Dataset - for multi-dimensional time serise data.
    Wraps a multi-dimensional signal and provides sub-sequences of fixed length as data samples.
    The signal is stored in a multi-dimensional torch tensor.

    :param signal: multi-dimensional sequential data, a tensor of at least two dimensions where the 
        first dimension is the time dimension and other dimensions represent the features.
    :param seqlen: sequence of this length are extracted from the signal.
    :param squeeze_label: if True, squeeze the labels at dimension 0.

    .. note:: 
        * Underlying signal is stored as ``torch.Tensor``
        * ``pandas.DataFrame`` object is used for disk IO
        * Call :func:`~known.ktorch.data.SeqDataset.dataloader` to get a torch Dataloader instance
        * We may want to squeeze the labels when connecting a dense layer towards the end of a model. Squeezing labels sets appropiate shape of `predicted` argument for a loss functions.
    """

    def __init__(self, signal:Tensor, seqlen:int, squeeze_label:bool=False) -> None:
        r""" 
        :param signal: multi-dimensional sequential data, a tensor of at least two dimensions where the 
            first dimension is the time dimension and other dimensions represent the features.
        :param seqlen: sequence of this length are extracted from the signal.
        :param squeeze_label: if `True`, squeeze the labels at dimension 0.
        """
        super().__init__()
        assert(signal.ndim>1), f'Must have at least two dimension'
        assert(signal.shape[0]>0), f'Must have at least one sample'
        self.signal = signal
        self.seqlen = seqlen
        self.features = signal.shape[1]
        self.count = len(self.signal)
        self.length = self.count - self.seqlen
        self.squeeze_label=squeeze_label
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        x = self.signal[index : index+self.seqlen]
        y = self.signal[index+self.seqlen : index+self.seqlen+1]
        if self.squeeze_label: y = y.squeeze(0)
        return x, y

    def dataloader(self, batch_size:int=1, shuffle:bool=None) -> DataLoader:
        r""" Returns a Dataloader object from this Dataset """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def read_csv(csv:str, cols:Iterable[str], reverse:bool, normalize:bool=False) -> np.ndarray:
        r""" 
        Reads data from a csv file using ``pandas.read_csv`` call. 
        The csv file is assumed to have time serise data in each of its columns i.e. 
        the columns are the features of the signal where as each row is a sample.

        :param csv:         path of csv file to read from
        :param cols:        name of columns to take features from
        :param reverse:     if `True`, reverses the sequence
        :param normalize:   if `True`, normalizes the data using min-max scalar

        :returns: time serise signal as ``ndarray``
        """
        df = pd.read_csv(csv) 
        C = []

        for c in cols:
            signal = df[c].to_numpy()
            if reverse: signal = signal[::-1]
            if normalize:
                scalar = np.maximum(np.abs(np.max(signal)), np.abs(np.min(signal)))
                if scalar!=0: signal=signal/scalar
            C.append(signal)
        return np.transpose(np.stack(C))

    @staticmethod
    def from_csv(csv:str, cols:Iterable[str], seqlen:int, reverse:bool, normalize:bool=False, 
                squeeze_label:bool=False, dtype=None) -> Dataset:
        r"""
        Creates a dataset instance from a csv file using :func:`~known.ktorch.data.SeqDataset.read_csv`.

        :param csv:         path of csv file to read from
        :param cols:        name of columns to take features from
        :param seqlen:      sequence of this length are extracted from the signal.
        :param reverse:     if `True`, reverses the sequence
        :param normalize:   if `True`, normalizes the data using min-max scalar
        :param squeeze_label: if `True`, squeeze the labels at dimension 0.
        :param dtype:       ``torch.dtype``

        :returns: an instance of this class

        .. note:: 
            This functions calls :func:`~known.ktorch.data.SeqDataset.read_csv`

        .. seealso:: 
            :func:`~known.ktorch.data.SeqDataset.to_csv`
        """
        return __class__(tt.from_numpy(
                np.copy(  __class__.read_csv(
                csv, cols, reverse, normalize=normalize) )).to(dtype=dtype), seqlen, squeeze_label=squeeze_label)

    @staticmethod
    def to_csv(ds, csv:str, cols:Iterable[str]):
        r"""
        Writes a dataset to a csv file using the ``pandas.DataFrame.to_csv`` call.
        
        :param csv:     path of csv file to write to
        :param cols:    column names to be used for each feature dimension

        .. seealso:: 
            :func:`~known.ktorch.data.SeqDataset.from_csv`
        """
        signal = np.transpose(ds.signal.numpy())
        df = pd.DataFrame({col:sig for sig,col in zip(signal, cols) }  )
        df.to_csv(csv)
        return

    @staticmethod
    def split_csv(csv:str, cols:Iterable[str], splits:Iterable[float], file_names:Iterable[str])-> Tuple:
        r"""
        Splits a csv according to provided ratios and saves each split to a seperate file.
        This can be used to split an existing csv dataset into train, test and validation sets.

        :param csv:         path of csv file to split
        :param cols:        name of columns to take features from
        :param splits:      ratios b/w 0 to 1 indicating size of splits
        :param file_names:  file names (full paths) used to save the splits

        :returns:
            * `siglen`: length of signal that was read from input csv
            * `splits`: same as provided in the argument
            * `splits_indices`: size of each split, indices of the signal that were the splitting points (start, end)
            * `split_sum`: sum of all splits, should be less than or equal to signal length

        .. seealso:: 
            :func:`~known.ktorch.data.SeqDataset.auto_split_csv`
        """
        assert len(splits)==len(file_names), f'splits and file_names must be of equal size'
        signal = __class__.read_csv(csv, cols, False)
        siglen = len(signal)
        split_sum=0
        splits_indices = []
        s=0
        for ratio,file_name in zip(splits,file_names):
            e=int(ratio*siglen)+s
            signal_slice = signal[s:e]
            splits_indices.append((s,e,e-s))
            split_sum+=(e-s)
            s=e
            df = pd.DataFrame({col: signal_slice[:,j] for j,col in enumerate(cols)})
            if (file_name is not None):
                if len(file_name)>0: df.to_csv(file_name)
        return siglen, splits, splits_indices, split_sum

    @staticmethod
    def auto_split_csv(csv:str, cols:Iterable[str], splits:Iterable[float], split_names:Union[Iterable[str], None]=None)-> Tuple:
        r"""
        Wraps ``split_csv``. 
        Produces splits in the same directory and auto generates the file names if not provided.

        :param csv:         path of csv file to split
        :param cols:        name of columns to take features from
        :param splits:      ratios b/w 0 to 1 indicating size of splits
        :param split_names: filename suffix used to save the splits

        .. note:: The splits are saved in the same directory as the csv file but they are suffixed with
            provided ``split_names``. For example, if ``csv='my_file.csv'`` and ``split_names=('train', 'test')``
            then two new files will be created named ``my_file_train.csv`` and ``my_file_test.csv``.
            In case ``split_names=None``, new files will have names ``my_file_1.csv`` and ``my_file_2.csv``.

        .. seealso:: 
            :func:`~known.ktorch.data.SeqDataset.split_csv`
        """
        if not split_names : split_names =   [ str(i+1) for i in range(len(splits)) ] 
        sepi = csv.rfind('.')
        fn, fe = csv[:sepi], csv[sepi:]
        file_names = [ f'{fn}_{sn}{fe}' for sn in split_names  ]
        return __class__.split_csv(csv=csv, cols=cols, splits=splits, file_names=file_names)

    @staticmethod
    def generate(csv:str, genF:Callable, colS:Iterable[str], normalize:bool=False, seed:Union[int, None]=None) -> pd.DataFrame: 
        r"""
        Generates a synthetic time serise dataset and saves it to a csv file

        :param csv: path of file to write to using ``pandas.DataFrame.to_csv`` call (should usually end with .csv)
        :param genF: a function like ``lambda rng, dim: y``,
            that generates sequential data at a given dimension ``dim``, where ``rng`` is a numpy RNG
        :param colS: column names for each dimension/feature
        :param normalize:   if True, normalizes the data using min-max scalar
        :param seed: seed for ``np.random.default_rng``
        
        :returns: ``pandas.DataFrame``

        .. note:: use ``DataFrame.info()`` and ``DataFrame.describe()`` to get information on generated data.

        .. seealso:: 
            :func:`~known.ktorch.data.SeqDataset.generateS`
        """
        rng = np.random.default_rng(seed)
        generated_priceS = []
        for dimS in range(len(colS)): generated_priceS.append(genF(rng, dimS))
        dd = {}
        for col,generated_price in zip(colS,generated_priceS):
            if normalize:
                scalar = np.maximum(np.abs(np.max(generated_price)), np.abs(np.min(generated_price)))
                if scalar!=0: generated_price/=scalar
            dd[col] = generated_price
        
        df = pd.DataFrame(dd)
        if (csv is not None):
            if len(csv)>0: df.to_csv(csv)
        return df 

    @staticmethod
    def generateS(csv:str, genF:Callable, colS:Iterable[str], normalize:bool=False, seed:Union[int, None]=None,
                splits:Iterable[float]=None, split_names:Union[Iterable[str], None]=None) -> Union[Tuple,pd.DataFrame]: 
        r"""
        Generate a synthetic time serise dataset with splits and save to csv files

        :returns: ``pandas.DataFrame`` if no spliting is performed else returns the outputs from ``split_csv``

        .. note:: This is the same as calling :func:`~known.ktorch.data.SeqDataset.generate` and :func:`~known.ktorch.data.SeqDataset.auto_split_csv` one after other. 
            If ``splits`` arg is `None` then no splitting is performed.

        .. seealso:: 
            :func:`~known.ktorch.data.SeqDataset.generate`
        """
        df = __class__.generate(csv, genF, colS, normalize, seed)
        if splits is not None:
            return __class__.auto_split_csv(csv, colS, splits, split_names)
        else:
            return df


class QuantiyMonitor:
    """ Monitors a quantity overtime to check if it improves (decreases) after a given patience. 
    Quantity is checked on each call to :func:`~known.ktorch.utils.QuantiyMonitor.check`. 
    The ``__call__`` methods implements the ``check`` method. Can be used to monitor loss for early stopping.
    
    :param name: name of the quantity to be monitored
    :param patience: number of calls before the monitor decides to stop
    :param delta: the amount by which the monitored quantity should decrease to consider an improvement
    """

    def __init__(self, name:str, patience:int, delta:float) -> None:
        r"""
        :param name: name of the quantity to be monitored
        :param patience: number of calls before the monitor decides to stop
        :param delta: the amount by which the monitored quantity should decrease to consider an improvement
        """
        assert(patience>0) # patience should be positive
        assert(delta>0) # delta should be positive
        self.patience, self.delta = patience, delta
        self.name = name
        self.reset()
    
    def reset(self, initial=None):
        r""" Resets the monitor's state and starts at a given `initial` value """
        self.last = (tt.inf if initial is None else initial)
        self.best = self.last
        self.counter = 0
        self.best_epoch = -1

    def __call__(self, current, epoch=-1, verbose=False) -> bool:
        return self.check(current, epoch, verbose)
        
    def check(self, current, epoch=-1, verbose=False) -> bool:
        r""" Calls the monitor to check the current value of monitored quality
        
        :param current: the current value of quantity
        :param epoch:   optional, the current epoch (used only for verbose)
        :param verbose: if `True`, prints monitor status when it changes

        :returns: `True` if the quanity has stopped improving, `False` otherwise.
        """
        self.last = current
        if self.best == tt.inf: 
            self.best=self.last
            self.best_epoch = epoch
            if verbose: print(f'|~|\t{self.name} Set to [{self.best}] on epoch {epoch}')
        else:
            delta = self.best - current # `loss` has decreased if `delta` is positive
            if verbose:
                if delta==0:
                    print(f'|~|\t{self.name} No Change(*) on epoch {epoch}')
                else:
                    if delta>0:
                        print(f'|~|\t{self.name} Decreased(-) by [{delta}] on epoch {epoch}')
                    else:
                        print(f'|~|\t{self.name} Increased(+) by [{-delta}] on epoch {epoch}')
            if delta > self.delta:
                # loss decresed more than self.delta
                if verbose: print(f'|~|\t{self.name} :DEC: [{delta} > {self.delta}] on epoch {epoch}') # {self.best} --> {current}, 
                self.best = current
                self.best_epoch = epoch
                self.counter = 0
            else:
                # loss didnt decresed more than self.delta
                self.counter += 1
                if self.counter >= self.patience: 
                    if verbose: print(f'|~| Stopping on {self.name} = [{current}] @ epoch {epoch} | best value = [{self.best}] @ epoch {self.best_epoch}')
                    return True # end of patience
        return False

class Trainer:
    r""" Holds a model, compiles it and trains/tests/evaluates it multiple times """
    
    def __init__(self, model) -> None:
        self.model = model

    def compile(self, criterionF, criterionA, optimizerF, optimizerA, lrsF, lrsA):
        if criterionA is None: criterionA={}
        if optimizerA is None: optimizerA={}
        if lrsA is None: lrsA={}
        self.criterion = criterionF(**criterionA)
        self.optimizer = optimizerF(self.model.parameters(), **optimizerA)
        self.lrs = lrsF(self.optimizer, **lrsA)
        
        


    def on_training_start(self, epochs):
        # can create a lr_scheduler here
        pass

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        # can step lr_scheduler, self.early_stop, chekpoint
        pass

    def on_training_end(self, epochs):
        pass

    def on_early_stop(self, epoch):
        pass

    def fit_epoch(self, data_loader):
        self.model.train()
        batch_loss=[]
        data_iter = iter(data_loader)
        while True:
            try:
                X, Y = next(data_iter)
                self.optimizer.zero_grad()
                P = self.model(X)
                #if P.shape!=Y.shape: print(f'!!!! {P.shape}, {Y.shape}')
                loss = self.criterion(P, Y)
                loss.backward()
                self.optimizer.step()
                loss_value = loss.item()
                batch_loss.append(loss_value)
            except StopIteration:
                break
        return np.array(batch_loss)

    @tt.no_grad()
    def eval_epoch(self, data_loader):
        self.model.eval()
        batch_loss=[]
        data_iter = iter(data_loader)
        while True:
            try:
                X, Y = next(data_iter)
                P = self.model(X)
                #if P.shape!=Y.shape: print(f'!!!! {P.shape}, {Y.shape}')
                loss_value = self.criterion(P, Y).item()
                batch_loss.append(loss_value)
            except StopIteration:
                break
        return np.array(batch_loss)



    def fit(self,
            training_data, validation_data,
            epochs,
            batch_size,
            shuffle,
            validation_freq,
            save_path,
            verbose=0
            ):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # assertions
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        assert self.model is not None, f'Model not available'
        assert self.criterion is not None, f'Criterion not available'
        assert self.optimizer is not None, f'Optimizer not available'
        assert training_data is not None, f'Training data not provided'
        assert epochs>0, f'Epochs should be at least 1'
        assert batch_size>0, f'Batch Size should be at least 1'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # additional checks and flags
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        do_validation = ((validation_freq>0) and (validation_data is not None))
        if validation_data is not None: 
            if validation_freq<=0: 
                print(f'[!] Validation data is provided but frequency is not set, Validation will not be performed')
                validation_freq=1
        else:
            if validation_freq>0: 
                print(f'[!] Validation frequency is set but data is not provided, Validation will not be performed')
                validation_freq=1

        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Data
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
        if verbose: 
            print(f'Training samples: [{len(training_data)}]')
            print(f'Training batches: [{len(training_data_loader)}]')

        if do_validation: 
            validation_data_loader = DataLoader(validation_data, batch_size=len(validation_data), shuffle=False)
            if verbose: 
                print(f'Validation samples: [{len(validation_data)}]')
                print(f'Validation batches: [{len(validation_data_loader)}]')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Temporary Variables
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        self.early_stop = False
        self.train_loss_history = []
        self.val_loss_history = []
        #self.save_path = save_path


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Training Loop
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        start_time=now()
        if verbose: 
            print('Start Training @ {}'.format(start_time))
            print('-------------------------------------------')
        self.on_training_start(epochs)
        for epoch in range(1, epochs+1):
            if verbose>1: print(f'[+] Epoch {epoch} of {epochs}')
            self.on_epoch_start(epoch)
            self.train_loss= self.fit_epoch(training_data_loader)
            self.train_loss_history.append(self.train_loss)
            self.mean_train_loss = np.mean(self.train_loss)
            if self.lrs is not None: self.lrs.step() # step learn rate
            if verbose>1: print(f'(-)\tTraining Loss: {self.mean_train_loss}')
            

            if do_validation and (epoch%validation_freq==0):
                #self.on_val_begin(epoch)
                self.val_loss = self.eval_epoch(validation_data_loader) 
                self.val_loss_history.append(self.val_loss)
                self.mean_val_loss = np.mean(self.val_loss)
                if verbose>1: print(f'(-)\tValidation Loss: {self.mean_val_loss}')
                #self.on_val_end(epoch)

            self.on_epoch_end(epoch)
            if self.early_stop: 
                if verbose: print(f'[~] Early-Stopping on epoch {epoch}')
                self.on_early_stop(epoch)
                break
        # end for epochs...................................................
        self.on_training_end(epochs)
        if save_path: 
            tt.save( self.model.state_dict(), save_path)
            if verbose: print(f'[*] Saved@ {save_path}')
        if verbose: print('-------------------------------------------')
        end_time=now()
        if verbose:
            print(f'Final Training Loss: [{np.mean(self.train_loss_history[-1])}]')
            if do_validation: print(f'Final Validation Loss: [{np.mean(self.val_loss_history[-1])}]') 
            print('End Training @ {}, Elapsed Time: [{}]'.format(end_time, end_time-start_time))
        return

    def evaluate(self, testing_data, batch_size=None):
        if batch_size is None: batch_size=len(testing_data)
        testing_data_loader=DataLoader(testing_data, batch_size=batch_size, shuffle=False)
        print(f'Testing samples: [{len(testing_data)}]')
        print(f'Testing batches: [{len(testing_data_loader)}]')
        test_loss = self.eval_epoch(testing_data_loader)
        mean_test_loss = np.mean(test_loss)
        print(f'Testing Loss: {mean_test_loss}') 
        return mean_test_loss, test_loss

    def plot_results(self, color='black', loss_plot_start=0):
        plt.figure(figsize=(12,6))
        plt.title('Training Loss')
        plt.plot(np.mean(self.train_loss_history,axis=1)[loss_plot_start:],color=color, label='train_loss')
        plt.legend()
        #plt.show()
        plt.close()
        if self.val_loss_history:
            plt.figure(figsize=(12,6))
            plt.title('Validation Loss')
            plt.plot(np.mean(self.val_loss_history,axis=1),color=color, label='val_loss', linestyle='dotted')
            plt.legend()
            #plt.show()
            plt.close()
        return
    def save_results(self, path:str):
        if not path.lower().endswith('.npz'): path = path + '.npz'
        np.savez(path, 
                 train_loss_history=np.array(self.train_loss_history), 
                 val_loss_history=np.array(self.val_loss_history))
        return