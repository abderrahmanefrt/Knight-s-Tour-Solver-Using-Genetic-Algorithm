import random
import tkinter as tk


class Chromosome:
  
   def __init__(self, genes=None):
       if genes is None:
            self.genes = [random.randint(1, 8) for _ in range(63)]
       else:
           self.genes=genes

   
   def crossover (self,partner):
       
       point=random.randint(1, 62)

       genes1=self.genes[:point]+ partner.genes[point:]
       genes2=partner.genes[:point]+ self.genes[point:]

       return Chromosome(genes1), Chromosome(genes2)
       

   def mutation(self, mutation_rate=0.01):
       for i in range(len(self.genes)):
           if random.random()< mutation_rate:
               self.genes[i]= random.randint(1,8)

       
class Knight:
    def __init__(self,chromosome=None):
        
        if chromosome is None:
            self.chromosome = Chromosome()
        else:
             self.chromosome = chromosome

        self.position= (0,0)
        self.path= [self.position]
        self.fitness=0

         

    def move_forward(self, direction):
     x, y = self.position

     if direction == 1:
         dx, dy = 1, -2
     elif direction == 2:
        dx, dy = 2, -1
     elif direction == 3:
        dx, dy = 2, 1
     elif direction == 4:
        dx, dy = 1, 2
     elif direction == 5:
        dx, dy = -1, 2
     elif direction == 6:
        dx, dy = -2, 1
     elif direction == 7:
        dx, dy = -2, -1
     elif direction == 8:
        dx, dy = -1, -2
     else:
         dx, dy = 0, 0  

     self.position = (x + dx, y + dy)

     self.path.append(self.position)


    def move_backward(self, direction):
     x, y = self.position

     if direction == 1:
        dx, dy = -1, 2
     elif direction == 2:
        dx, dy = -2, 1
     elif direction == 3:
        dx, dy = -2, -1
     elif direction == 4:
        dx, dy = -1, -2
     elif direction == 5:
        dx, dy = 1, -2
     elif direction == 6:
        dx, dy = 2, -1
     elif direction == 7:
        dx, dy = 2, 1
     elif direction == 8:
        dx, dy = 1, 2

     self.position = (x + dx, y + dy)

     if len(self.path) > 1:
        self.path.pop()


    def check_moves(self):
        self.position = (0, 0)
        self.path = [self.position]
        
        cycle_forward = random.choice([True, False])
        
        # Parcourir chaque gène (mouvement) dans le chromosome
        for gene in self.chromosome.genes:
            original_move = gene
            move_found = False
            
            # Essayer le mouvement original
            self.move_forward(original_move)
            
            # Vérifier si la nouvelle position est valide
            x, y = self.position
            if 0 <= x < 8 and 0 <= y < 8 and self.position not in self.path[:-1]:
                move_found = True
            else:
                # Annuler le mouvement invalide
                self.move_backward(original_move)
                
                # Tester les autres mouvements en cycle
                for i in range(1, 8):
                    if cycle_forward:
                        new_move = ((original_move + i - 1) % 8) + 1
                    else:
                        new_move = ((original_move - i - 1) % 8) + 1
                    
                    self.move_forward(new_move)
                    x, y = self.position
                    
                    if 0 <= x < 8 and 0 <= y < 8 and self.position not in self.path[:-1]:
                        move_found = True
                        break
                    else:
                        self.move_backward(new_move)
                
                # Si aucun mouvement valide n'est trouvé, garder le dernier mouvement
                if not move_found:
                    self.move_forward(original_move)



    def evaluate_fitness(self):
      self.fitness = len(set(self.path))
      return self.fitness
    
class Population:
    def __init__(self, population_size):
        self.population_size = population_size
        self.generation = 1
        self.knights = [Knight() for _ in range(population_size)]

    def check_population(self):
        for knight in self.knights:
            knight.check_moves()

    def evaluate(self):
        best_knight = None
        max_fitness = 0
        
        for knight in self.knights:
            fitness = knight.evaluate_fitness()
            if fitness > max_fitness:
                max_fitness = fitness
                best_knight = knight
        
        return max_fitness, best_knight

    def tournament_selection(self, size=3):
        sample = random.sample(self.knights, size)
        sample.sort(key=lambda k: k.fitness, reverse=True)
        return sample[0], sample[1]

    def create_new_generation(self):
        new_knights = []
        
        while len(new_knights) < self.population_size:
            parent1, parent2 = self.tournament_selection()
            child1_genes, child2_genes = parent1.chromosome.crossover(parent2.chromosome)
            
            child1_genes.mutation()
            child2_genes.mutation()
            
            new_knights.append(Knight(child1_genes))
            if len(new_knights) < self.population_size:
                new_knights.append(Knight(child2_genes))
        
        self.knights = new_knights
        self.generation += 1



def visualize_solution(knight):
    root = tk.Tk()
    root.title(f"Knight's Tour Solution (Fitness: {knight.fitness})")
    
    canvas_size = 640
    square_size = canvas_size // 8
    
    canvas = tk.Canvas(root, width=canvas_size, height=canvas_size)
    canvas.pack(padx=10, pady=10)
    
    # Dessiner l'échiquier
    for row in range(8):
        for col in range(8):
            color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
            canvas.create_rectangle(
                col * square_size, row * square_size,
                (col + 1) * square_size, (row + 1) * square_size,
                fill=color, outline="black"
            )
    
    # Dessiner le chemin du chevalier
    for i in range(len(knight.path) - 1):
        x1, y1 = knight.path[i]
        x2, y2 = knight.path[i + 1]
        
        center_x1 = x1 * square_size + square_size // 2
        center_y1 = y1 * square_size + square_size // 2
        center_x2 = x2 * square_size + square_size // 2
        center_y2 = y2 * square_size + square_size // 2
        
        canvas.create_line(center_x1, center_y1, center_x2, center_y2,
                          fill="red", width=2, arrow=tk.LAST)
    
    # Dessiner les numéros de positions
    for i, (x, y) in enumerate(knight.path):
        center_x = x * square_size + square_size // 2
        center_y = y * square_size + square_size // 2
        
        canvas.create_oval(center_x - 15, center_y - 15,
                          center_x + 15, center_y + 15,
                          fill="white", outline="black", width=2)
        canvas.create_text(center_x, center_y, text=str(i + 1),
                          font=("Arial", 10, "bold"))
    
    root.mainloop()


def main():
    population_size = 50
    
    # Create the initial population
    population = Population(population_size)
    
    print("Starting Genetic Algorithm for Knight's Tour...")
    
    while True:
        # Check the validity of the current population
        population.check_population()
        
        # Evaluate the current generation and get the best knight with its fitness value
        maxFit, bestSolution = population.evaluate()
        
        print(f"Generation {population.generation}: Best Fitness = {maxFit}")
        
        if maxFit == 64:
            print(f"\nSolution found in generation {population.generation}!")
            break
        
        # Limite optionnelle pour éviter une boucle infinie
        if population.generation >= 1000:
            print(f"\nStopped after {population.generation} generations.")
            print(f"Best fitness achieved: {maxFit}")
            break
        
        # Generate the new population
        population.create_new_generation()
    
    # Create the user interface to display the solution
    visualize_solution(bestSolution)


if __name__ == "__main__":
    main()