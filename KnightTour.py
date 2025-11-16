import random

# -------------------------
# Chromosome
# -------------------------
class Chromosome:
    """
    Chromosome : liste de 63 gènes (valeurs 1..8 représentant les 8 mouvements du cavalier)
    """
    LENGTH = 63

    def __init__(self, genes=None):
        if genes is None:
            self.genes = [random.randint(1, 8) for _ in range(self.LENGTH)]
        else:
            if len(genes) != self.LENGTH:
                raise ValueError(f"Genes length must be {self.LENGTH}")
            self.genes = genes

    def crossover(self, partner, crossover_prob=1.0):
        """
        Single-point crossover (conforme au cours).
        Si probabilité de crossover < random, on retourne des copies des parents.
        Retourne deux Chromosome (enfants).
        """
        if random.random() > crossover_prob:
            return Chromosome(self.genes[:]), Chromosome(partner.genes[:])

        point = random.randint(1, self.LENGTH - 1)
        child1_genes = self.genes[:point] + partner.genes[point:]
        child2_genes = partner.genes[:point] + self.genes[point:]
        return Chromosome(child1_genes), Chromosome(child2_genes)

    def mutation(self, mutation_prob=0.01):
        """
        Flip mutation : chaque gène a une chance mutation_prob d'être remplacé
        par une valeur aléatoire dans 1..8 (conforme au cours).
        """
        for i in range(len(self.genes)):
            if random.random() < mutation_prob:
                self.genes[i] = random.randint(1, 8)


# -------------------------
# Knight
# -------------------------
class Knight:
    """
    Représente un individu : position courante, chromosome (séquence de mouvements),
    path (positions successives), fitness (nombre de cases valides visitées).
    """

    # mapping directions -> (dx, dy)
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
            # Accept either Chromosome instance or raw genes list
            if isinstance(chromosome, Chromosome):
                self.chromosome = chromosome
            else:
                self.chromosome = Chromosome(chromosome)

        self.position = (0, 0)
        self.path = [self.position]
        self.fitness = 0

    def move_forward(self, direction):
        """Applique le déplacement direction et ajoute la nouvelle position à path."""
        x, y = self.position
        dx, dy = Knight.MOVES.get(direction, (0, 0))
        new_pos = (x + dx, y + dy)
        self.position = new_pos
        self.path.append(new_pos)

    def move_backward(self, direction):
        """
        Annule le dernier déplacement correspondant à 'direction'.
        On suppose qu'on annule immédiatement le dernier append fait par move_forward.
        """
        # Revenir à la position précédente (si possible)
        if len(self.path) > 1:
            # retirer la dernière position
            self.path.pop()
            # mettre à jour position sur l'avant-dernière
            self.position = self.path[-1]
        else:
            # Si on est au début, on reste à (0,0)
            self.position = (0, 0)
            self.path = [self.position]

    def check_moves(self):
        """
        Parcourt chaque gène et tente d'appliquer le mouvement.
        Si le mouvement est illégal (hors échiquier ou case déjà visitée),
        on annule et on tente les autres mouvements dans l'ordre cyclique
        (forward ou backward), la direction de cycle est choisie aléatoirement
        et reste la même pour tout le chromosome.
        Si aucun mouvement valide n'est trouvé, on garde le dernier mouvement
        (comme demandé dans l'énoncé) — cela peut produire une position
        invalide qui sera ensuite comptée par evaluate_fitness.
        """
        # reset position and path
        self.position = (0, 0)
        self.path = [self.position]

        cycle_forward = random.choice([True, False])  # choisi une fois par chromosome

        for gene in self.chromosome.genes:
            original_move = gene
            move_found = False

            # Essayer le mouvement original
            self.move_forward(original_move)
            x, y = self.position

            # Vérifier validité : bord et non-visité (excluant la dernière position ajoutée)
            if 0 <= x < 8 and 0 <= y < 8 and self.position not in self.path[:-1]:
                move_found = True
            else:
                # annuler le mouvement invalide
                self.move_backward(original_move)

                # tester les 7 autres mouvements selon le cycle
                for i in range(1, 8):
                    if cycle_forward:
                        new_move = ((original_move + i - 1) % 8) + 1
                    else:
                        # cycle backward
                        new_move = ((original_move - i - 1) % 8) + 1

                    self.move_forward(new_move)
                    x, y = self.position

                    if 0 <= x < 8 and 0 <= y < 8 and self.position not in self.path[:-1]:
                        move_found = True
                        break
                    else:
                        # annuler et continuer
                        self.move_backward(new_move)

                # Si aucun mouvement valide n'est trouvé, garder le dernier mouvement original
                if not move_found:
                    # On applique le mouvement original (même s'il est invalide)
                    self.move_forward(original_move)

    def evaluate_fitness(self):
        """
        Parcourt la liste self.path et compte le nombre de positions valides successives
        (à partir de la première) jusqu'à rencontrer une position invalide
        (hors échiquier ou déjà visitée). La fitness maximale est 64.
        """
        seen = set()
        fitness = 0

        for pos in self.path:
            x, y = pos
            # vérifier bord
            if not (0 <= x < 8 and 0 <= y < 8):
                break
            # vérifier répétition
            if pos in seen:
                break
            seen.add(pos)
            fitness += 1
            if fitness >= 64:
                break

        self.fitness = fitness
        return self.fitness


# -------------------------
# Population
# -------------------------
class Population:
    """
    Population contenant N knights. Fournit les méthodes demandées dans l'énoncé.
    """

    def __init__(self, population_size, mutation_prob=0.01, tournament_size=3):
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.generation = 1
        self.knights = [Knight() for _ in range(population_size)]

        # Évaluer initialement (on peut laisser la check_population() faire la correction d'abord)
        for k in self.knights:
            # Par défaut on ne corrige pas automatiquement; l'appel check_population() le fera.
            k.evaluate_fitness()

    def check_population(self):
        """Pour chaque chevalier, vérifier et corriger ses mouvements."""
        for knight in self.knights:
            knight.check_moves()

    def evaluate(self):
        """
        Évalue la fitness de tous les individus et retourne (max_fitness, best_knight).
        """
        best = None
        max_fit = -1
        for knight in self.knights:
            fit = knight.evaluate_fitness()
            if fit > max_fit:
                max_fit = fit
                best = knight
        return max_fit, best

    def tournament_selection(self, size=None):
        """
        Sélection par tournoi : on échantillonne 'size' individus aléatoirement
        et on renvoie les deux meilleurs d'entre eux.
        """
        if size is None:
            size = self.tournament_size
        sample = random.sample(self.knights, size)
        sample.sort(key=lambda k: k.fitness, reverse=True)
        # retourner parent1, parent2
        return sample[0], sample[1]

    def create_new_generation(self, crossover_prob=1.0):
        """
        Crée une nouvelle génération en respectant :
        - sélection par tournoi (taille self.tournament_size)
        - crossover single-point (probabilité crossover_prob)
        - mutation (self.mutation_prob)
        Remplace complètement la population (génération complète).
        """
        new_knights = []

        while len(new_knights) < self.population_size:
            parent1, parent2 = self.tournament_selection()
            # crossover retourne deux Chromosome
            child_chrom1, child_chrom2 = parent1.chromosome.crossover(parent2.chromosome, crossover_prob)

            # mutation
            child_chrom1.mutation(self.mutation_prob)
            child_chrom2.mutation(self.mutation_prob)

            # créer Knight à partir des Chromosome
            k1 = Knight(child_chrom1)
            k2 = Knight(child_chrom2)

            # ajouter à la nouvelle population
            new_knights.append(k1)
            if len(new_knights) < self.population_size:
                new_knights.append(k2)

        self.knights = new_knights
        self.generation += 1


# -------------------------
# MAIN (console-only)
# -------------------------
def main():
    random.seed()  # tu peux fixer une seed pour reproductibilité : ex random.seed(42)

    population_size = 50
    mutation_prob = 0.01
    tournament_size = 3
    max_generations = 1000

    population = Population(population_size, mutation_prob, tournament_size)

    print("Démarrage de l'algorithme génétique pour le Knight's Tour (console-only)")
    print("Population size:", population_size)
    print("Mutation prob:", mutation_prob)
    print("Tournament size:", tournament_size)
    print("-------------------------------------------")

    while True:
        # Correction des chromosomes (check_moves)
        population.check_population()

        # Évaluation
        maxFit, bestSolution = population.evaluate()
        print(f"Generation {population.generation} | Best Fitness = {maxFit}")

        # Condition de réussite
        if maxFit == 64:
            print(f"\nSolution trouvée en génération {population.generation} !")
            break

        # Condition d'arrêt pour éviter boucle infinie
        if population.generation >= max_generations:
            print(f"\nArrêt après {population.generation} générations. Meilleur fitness obtenu: {maxFit}")
            break

        # Génération suivante
        population.create_new_generation()

    # Affichage résumé final (console)
    best = bestSolution
    print("\n--- Résumé ---")
    print("Best fitness:", best.fitness)
    print("Path length:", len(best.path))
    print("Path (positions):", best.path)
    print("Genes:", best.chromosome.genes)


if __name__ == "__main__":
    main()
