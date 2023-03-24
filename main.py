import random

class GeneticAlgorithm:
    def __init__(self, max_gen=200, pop_dim=100,  pc = 0.9, pm=0.2):
        self.counter = 0 #Gerações
        self.max_gen = max_gen
        self.pop_size = pop_dim
        self.pm = pm
        self.pc = pc
        self.par = 2
        self.nbits = 8
        self.lchrome = self.npar * self.nbits
        self.vmax = [1,1]
        self.vmin = [0,0]
