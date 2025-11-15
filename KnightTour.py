import random


class Chromosome:
   
   
   def __init__(self, genes):
       if genes is None:
           genes= random.sample(range(64),64)
       else:
           self.genes=genes
    

      