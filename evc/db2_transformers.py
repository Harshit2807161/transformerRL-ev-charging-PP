import pickle

class PricePredictor:
    def __init__(self, path, past, future, auto_load=True):
        self.past, self.future = past, future
        self.past = 24
        self.future = 1
        self.path = path
        self.model = None
        
        if auto_load:
            self.load()
    
    def load(self): 
        # Load the object from the file
        with open(self.path, 'rb') as file:
            self.model = pickle.load(file)

    def __call__(self, index):
        return self.model[index]
