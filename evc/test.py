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

lr = 0.00004
lref = .25

start_lr, end_lr = (float(lr)+0.00001), (float(lr)+0.00001)*(float(lref)+0.35)      # select-arg
lr_mapper=REMAP((-0.2,1), (start_lr, end_lr)) # set learn rate schedluer

def lr_schedule(progress): return lr_mapper.in2map(1-progress) #lr

print("Init: ",lr_schedule(1))
print("Fin: ",lr_schedule(0))