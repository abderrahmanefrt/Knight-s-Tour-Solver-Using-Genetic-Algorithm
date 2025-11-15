import random


class Chromosome:
  
   def __init__(self, genes):
       if genes is None:
            self.genes = [random.randint(1, 8) for _ in range(63)]
       else:
           self.genes=genes

   
   def crossover (self,partner):
       
       point=random.randint(1, 62)

       genes1=self.gens[:point]+ partner.genes[point:]
       genes2=partner.genes[:point]+ self.genes[point:]

       return Chromosome(genes1), Chromosome(genes2)
       

   def mutation(self,mutation_rate ):
       for i in range(len(self.genes)):
           if random.random()< mutation_rate:
               self.genes[i]= random.randint(1,8)

       
class Knightclass:
    position =(0,0)
    Chromosome= None
    path= []
    fitness= 0

    def __int__(self,Chromosome):
        
        self.Chromosome= Chromosome
        self.position= (0,0)
        self.path= [self.position]
        self.fitness=0
        
        
        