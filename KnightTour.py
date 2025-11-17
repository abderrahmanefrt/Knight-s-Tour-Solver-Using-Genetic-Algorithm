import random
import pygame
import sys

class Chromosome:
    LENGTH = 63

    def __init__(self, genes=None):
        if genes is None:
            self.genes = [random.randint(1, 8) for _ in range(self.LENGTH)]
        else:
            if len(genes) != self.LENGTH:
                raise ValueError(f"Genes length must be {self.LENGTH}")
            self.genes = genes[:]

    def crossover(self, partner, crossover_prob=1.0):
        if random.random() <= crossover_prob:
            crossover_point = random.randint(1, self.LENGTH - 1)
            offspring1_genes = self.genes[:crossover_point] + partner.genes[crossover_point:]
            offspring2_genes = partner.genes[:crossover_point] + self.genes[crossover_point:]
            return Chromosome(offspring1_genes), Chromosome(offspring2_genes)
        else:
            return Chromosome(self.genes), Chromosome(partner.genes)

    def mutation(self, mutation_prob=0.01):
        for i in range(len(self.genes)):
            if random.random() <= mutation_prob:
                self.genes[i] = random.randint(1, 8)


# Knight
class Knight:
    MOVES = {
        1: (1, -2),
        2: (2, -1),
        3: (2, 1),
        4: (1, 2),
        5: (-1, 2),
        6: (-2, 1),
        7: (-2, -1),  
        8: (-1, -2),
    }

    def __init__(self, chromosome=None):
        if chromosome is None:
            self.chromosome = Chromosome()
        else:
            if isinstance(chromosome, Chromosome):
                self.chromosome = chromosome
            else:
                self.chromosome = Chromosome(chromosome)

        self.position = (0, 0)
        self.path = [self.position]
        self.fitness = 0

    def move_forward(self, direction):
        x, y = self.position
        dx, dy = Knight.MOVES.get(direction, (0, 0))
        new_pos = (x + dx, y + dy)
        self.position = new_pos
        self.path.append(new_pos)

    def move_backward(self, direction):
        if len(self.path) > 1:
            self.path.pop()
            self.position = self.path[-1]
        else:
            self.position = (0, 0)
            self.path = [self.position]

    def check_moves(self):
        self.position = (0, 0)
        self.path = [self.position]
        cycle_forward = random.choice([True, False])

        for gene in self.chromosome.genes:
            original_move = gene
            move_found = False

            self.move_forward(original_move)
            x, y = self.position

            if 0 <= x < 8 and 0 <= y < 8 and self.position not in self.path[:-1]:
                move_found = True
            else:
                self.move_backward(original_move)

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

                if not move_found:
                    self.move_forward(original_move)

    def evaluate_fitness(self):
        seen = set()
        fitness = 0

        for pos in self.path:
            x, y = pos
            if not (0 <= x < 8 and 0 <= y < 8):
                break
            if pos in seen:
                break
            seen.add(pos)
            fitness += 1
            if fitness >= 64:
                break

        self.fitness = fitness
        return self.fitness


# Population
class Population:
    def __init__(self, population_size, mutation_prob=0.001, tournament_size=3, crossover_prob=1.0):
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.generation = 1
        self.knights = [Knight() for _ in range(population_size)]
        for knight in self.knights:
            knight.evaluate_fitness()

    def check_population(self):
        for knight in self.knights:
            knight.check_moves()

    def evaluate(self):
        best_knight = None
        max_fitness = -1
        for knight in self.knights:
            fit = knight.evaluate_fitness()
            if fit > max_fitness:
                max_fitness = fit
                best_knight = knight
        return max_fitness, best_knight

    def tournament_selection(self, size=None):
        if size is None:
            size = self.tournament_size
        sample = random.sample(self.knights, size)
        sample.sort(key=lambda k: k.fitness, reverse=True)
        return sample[0], sample[1]

    def create_new_generation(self):
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = self.tournament_selection()
            offspring_chrom1, offspring_chrom2 = parent1.chromosome.crossover(
                partner=parent2.chromosome,
                crossover_prob=self.crossover_prob
            )
            offspring_chrom1.mutation(self.mutation_prob)
            offspring_chrom2.mutation(self.mutation_prob)
            knight1 = Knight(offspring_chrom1)
            knight2 = Knight(offspring_chrom2)
            new_population.append(knight1)
            if len(new_population) < self.population_size:
                new_population.append(knight2)
        self.knights = new_population
        self.generation += 1



def visualize_with_pygame(knight, fitness, generation, params, title="Knight's Tour - GA", square_px=80):
  
    pygame.init()

    board_px = square_px * 8
    SIDE_PANEL_WIDTH = 300  
    STATS_HEIGHT = 100  
    
    window_width = board_px + SIDE_PANEL_WIDTH
    window_height = board_px + STATS_HEIGHT

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(title)

    # Couleurs
    WHITE = (255, 255, 255)
    GREEN = (34, 139, 34)
    BLACK = (0, 0, 0)
    RED = (200, 30, 30)
    GREY = (210, 210, 210)
    DARK_GREEN = (0, 100, 0)
    LIGHT_GREY = (240, 240, 240)
    BLUE = (50, 50, 200)

    font_small = pygame.font.SysFont("Arial", 18)
    font = pygame.font.SysFont("Arial", 22)
    font_medium = pygame.font.SysFont("Arial", 24)
    big_font = pygame.font.SysFont("Arial", 30)

    path = knight.path
    display_positions = [(x, y) for (x, y) in path if 0 <= x < 8 and 0 <= y < 8]

    def center_of(cell):
        x, y = cell
        return x * square_px + square_px // 2, y * square_px + square_px // 2

    # Animation variables
    step = 0
    paused = False
    running = True

    clock = pygame.time.Clock()
    speed_delay = 200 

    while running:
        screen.fill((230, 230, 230))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()

                if board_px + 50 <= mx <= board_px + 250 and 400 <= my <= 460:
                    paused = not paused

        for row in range(8):
            for col in range(8):
                color = GREEN if (row + col) % 2 == 0 else WHITE
                pygame.draw.rect(screen, color, (col * square_px, row * square_px, square_px, square_px))

        if not paused and step < len(display_positions):
            step += 1
            pygame.time.delay(speed_delay)

        points = []
        for i in range(step):
            x, y = display_positions[i]
            cx, cy = center_of((x, y))
            points.append((cx, cy))

            pygame.draw.circle(screen, WHITE, (cx, cy), square_px // 4)
            num = font_small.render(str(i + 1), True, BLACK)
            screen.blit(num, num.get_rect(center=(cx, cy)))

        if len(points) >= 2:
            pygame.draw.lines(screen, RED, False, points, 3)

        pygame.draw.rect(screen, LIGHT_GREY, (board_px, 0, SIDE_PANEL_WIDTH, window_height))

        panel_x = board_px + 20
        y_offset = 30
        
        title_panel = big_font.render("INFORMATIONS", True, DARK_GREEN)
        screen.blit(title_panel, (panel_x, y_offset))
        y_offset += 50

        pygame.draw.rect(screen, WHITE, (panel_x, y_offset, 260, 100), border_radius=10)
        
        result_title = font_medium.render("Résultats", True, BLUE)
        screen.blit(result_title, (panel_x + 10, y_offset + 10))
        
        txt_fitness = font.render(f"Fitness : {fitness}/64", True, BLACK)
        txt_gen = font.render(f"Génération : {generation}", True, BLACK)
        
        screen.blit(txt_fitness, (panel_x + 10, y_offset + 45))
        screen.blit(txt_gen, (panel_x + 10, y_offset + 70))
        
        y_offset += 120

        pygame.draw.rect(screen, WHITE, (panel_x, y_offset, 260, 220), border_radius=10)
        
        params_title = font_medium.render("Paramètres", True, BLUE)
        screen.blit(params_title, (panel_x + 10, y_offset + 10))
        
        param_y = y_offset + 45
        line_height = 30
        
        txt_pop = font_small.render(f"Population : {params['population_size']}", True, BLACK)
        txt_mut = font_small.render(f"Mutation : {params['mutation_prob']}", True, BLACK)
        txt_cross = font_small.render(f"Crossover : {params['crossover_prob']}", True, BLACK)
        txt_tourn = font_small.render(f"Tournament : {params['tournament_size']}", True, BLACK)
        txt_maxgen = font_small.render(f"Max gen. : {params['max_generations']}", True, BLACK)
        
        screen.blit(txt_pop, (panel_x + 15, param_y))
        screen.blit(txt_mut, (panel_x + 15, param_y + line_height))
        screen.blit(txt_cross, (panel_x + 15, param_y + line_height * 2))
        screen.blit(txt_tourn, (panel_x + 15, param_y + line_height * 3))
        screen.blit(txt_maxgen, (panel_x + 15, param_y + line_height * 4))
        
        y_offset += 240

        btn_y = 400
        btn_color = GREY if not paused else GREEN
        pygame.draw.rect(screen, btn_color, (board_px + 50, btn_y, 200, 60), border_radius=15)
        
        label = "⏸  PAUSE" if not paused else "▶  PLAY"
        btn_txt = font_medium.render(label, True, WHITE if not paused else BLACK)
        btn_rect = btn_txt.get_rect(center=(board_px + 150, btn_y + 30))
        screen.blit(btn_txt, btn_rect)

        # --- ZONE DES STATS EN BAS (sous l'échiquier) ---
        pygame.draw.rect(screen, WHITE, (0, board_px, board_px, STATS_HEIGHT))
        
        status_txt = font.render(f"Animation : {'En pause' if paused else 'En cours'}  |  Étape : {step}/{len(display_positions)}", True, BLACK)
        screen.blit(status_txt, (20, board_px + 40))

        pygame.display.flip()
        clock.tick(12)

    pygame.quit()



def run_genetic_and_visualize(
    population_size=50,
    mutation_prob=0.001,
    tournament_size=3,
    crossover_prob=1.0,
    max_generations=1000,
    animate=True
):
    population = Population(
        population_size=population_size,
        mutation_prob=mutation_prob,
        tournament_size=tournament_size,
        crossover_prob=crossover_prob
    )

    
    print("Algorithme Génétique Knight's Tour")
  
    print(f"Population size  : {population_size}")
    print(f"Mutation probability  : {mutation_prob}")
    print(f"Crossover probability : {crossover_prob}")
    print(f"Tournament size : {tournament_size}")
    print(f"Max generations : {max_generations}")
    
    print()

    best_solution = None
    while True:
        population.check_population()
        maxFit, bestSolution = population.evaluate()
        best_solution = bestSolution

        if maxFit == 64:
            print("\nsolution trouve")
            break

        if population.generation >= max_generations:
            print("\nNombre max de générations atteint.")
            break

        population.create_new_generation()

    print()
    print(f"Fitness finale: {best_solution.fitness}/64")
    print(f"Longueur path: {len(best_solution.path)}")
    print(f"Genes (20 premiers): {best_solution.chromosome.genes[:20]}")
    print()

    params = {
        'population_size': population_size,
        'mutation_prob': mutation_prob,
        'crossover_prob': crossover_prob,
        'tournament_size': tournament_size,
        'max_generations': max_generations
     }

    visualize_with_pygame(best_solution, best_solution.fitness, population.generation,params)


if __name__ == "__main__":
    run_genetic_and_visualize(
        population_size=50,
        mutation_prob=0.001,
        tournament_size=3,
        crossover_prob=1.0,
        max_generations=1000,
        animate=True
    )

