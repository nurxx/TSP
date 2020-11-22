import numpy as np, pandas as pd, matplotlib.pyplot as plt
import random, operator
from collections import Counter
from math import factorial


class City:
    def __init__(self):
        self.x = int(random.random() * 200)
        self.y = int(random.random() * 200)
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
    
    # Calculate distance between two cities
    def evaluate_distance(self, city):
        xd = abs(self.x - city.x)
        yd = abs(self.y - city.y)
        return np.sqrt(xd**2 + yd**2)

class Route:
    def __init__(self, route:list):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
    
    # Calculate path length of route   
    def evaluate_route_distance(self) -> int:
        if self.distance == 0:
            path_distance = 0
            for i in range(len(self.route)):
                from_city = self.route[i]
                to_city = None
                if i + 1 < len(self.route):
                    to_city = self.route[i + 1]
                else:
                    to_city = self.route[0]
                path_distance += from_city.evaluate_distance(to_city)
            self.distance = path_distance
        return self.distance
    
    # Fitness function to evaluate how good a route is/how short the distance is
    def evaluate_route_fitness(self) -> float:
        if self.fitness == 0.0:
            self.fitness = 1.0 / float(self.evaluate_route_distance())
        return self.fitness
    
    def __len__(self):
        return len(self.route)
    
    def __getitem__(self, index):
        return self.route[index]
    
    def __setitem__(self, index, value):
        self.route[index] = value
    
    def __repr__(self):
        for index,city in enumerate(self.route):
            print(city)
            
    

# Create initial population - collection of routes by given population size
def get_initial_population(population_size, cities:list) -> list:
    routes = []
    for i in range(0, population_size):
        # random.sample(): Returns a new list containing elements from the population while leaving the original population unchanged.
        routes.append(Route(random.sample(cities, len(cities))))
    return routes

def sort_routes_by_fitness(routes:list) -> dict:
    population_size = len(routes)
    result = {}
    for i in range(0,population_size):
        result[i] = Route(routes[i]).evaluate_route_fitness()
    return sorted(result.items(), key = operator.itemgetter(1), reverse = True)

def select_survivors(routes:list, sorted_routes:dict, elite_size:int) -> list:
    survivors = []
    total_fitness = sum([Route(route).evaluate_route_fitness() for route in routes])
    
    # Keep fittest routes in order to move to the next generation
    for i in range(0, elite_size):
        survivors.append(sorted_routes[i][0])

    # Using fitness proportionate selection for the rest (aka. roulette wheel)
    for _ in range(0, len(sorted_routes) - elite_size):
        pick = random.uniform(0, total_fitness)
        current = 0
        for i in range(0, len(sorted_routes)):
            current += sorted_routes[i][1]
            if pick <= current :
                survivors.append(sorted_routes[i][0])
                break
    return survivors

# Get collection of parents that will be used to create the next generation
def get_mating_pool(routes:list, survivors:list) -> list:
    pool = []
    for i in range(0, len(survivors)):
        pool.append(routes[survivors[i]])
    return pool

# Crossover 2 routes
def crossover(parent1:list, parent2:list) -> list:
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    start_gene = min(geneA, geneB)
    end_gene = max(geneA, geneB)

    for i in range(start_gene, end_gene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

# Crossover all routes
def crossover_population(mating_pool:list, elite_size:int) -> list:
    children = []
    length = len(mating_pool) - elite_size
    pool = random.sample(mating_pool, len(mating_pool))

    for i in range(0, elite_size):
        children.append(mating_pool[i])
    
    for i in range(0, length):
        child = crossover(pool[i], pool[len(mating_pool)-i-1])
        children.append(child)
    return children

# Mutate route
def mutate(route:list, mutation_rate:float) -> list:
    for swapped in range(len(route)):
        if(random.random() < mutation_rate):
            swap_with = int(random.random() * len(route))
            
            city1 = route[swapped]
            city2 = route[swap_with]
            
            route[swapped] = city2
            route[swap_with] = city1
    return route

# Mutate all routes
def mutate_population(routes:list, mutation_rate:float) -> list:
    mutated = []
    for idx in range(0, len(routes)):
        index = mutate(routes[idx], mutation_rate)
        mutated.append(index)
    return mutated

# Create a new generation
def get_next_generation(routes:list, elite_size:int, mutation_rate:float) -> list:
    sorted_by_fitness = sort_routes_by_fitness(routes)
    survivors = select_survivors(routes, sorted_by_fitness, elite_size)
    
    mating_pool = get_mating_pool(routes, survivors)
    children = crossover_population(mating_pool, elite_size)
    next_generation = mutate_population(children, mutation_rate)
    
    return next_generation

# Check convergence by given rate
def check_convergence(population:list, convergence_rate:float) -> bool:
    fitness_values = [route[1] for route in sort_routes_by_fitness(population)]
    repeated_fitness_values = {value:fitness_values.count(value) for value in fitness_values}
    most_common = sorted(repeated_fitness_values.items(), key=lambda kv: kv[1], reverse=True)
    
    rate = most_common[0][1] / len(population) 
    # print(rate)
    return rate >= convergence_rate
    
def get_best_route(cities:list, population_size:int, elite_size:int, mutation_rate:float, convergence:int, generations = None) -> list:
    population = get_initial_population(population_size, cities)
    
    progress = {}
    best_current_fitness = sort_routes_by_fitness(population)[0][1]
    best_current_distance = sort_routes_by_fitness(population)[0][1]**(-1)
    print(f"Initial distance: {str(best_current_distance)}")
    
    convergence_rate = convergence
    current_generation = 0
    progress[current_generation] = best_current_distance
    while check_convergence(population, convergence_rate) is not True:
        population = get_next_generation(population, elite_size, mutation_rate)
        next_generation_fitness = sort_routes_by_fitness(population)[0][1]
        next_generation_distance = sort_routes_by_fitness(population)[0][1]**(-1)
        progress[current_generation] = next_generation_distance
        current_generation += 1
    
        
    print(f"Final distance: {str(next_generation_distance)}")
    progress[current_generation] = next_generation_distance
    best_route_index = sort_routes_by_fitness(population)[0][0]
    best_route = population[best_route_index]
     
    print_indeces = [10, current_generation]
    while len(print_indeces) != 5:
        idx = random.randint(11, current_generation)
        if idx not in print_indeces:
            print_indeces.append(idx)
    
    print("Printing progress...")
    for index in sorted(progress):
        if index in print_indeces:
            print(f"{index}th generation:\t{progress[index]}")
    
    p = list(progress.values())
    plt.plot(p)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
    
    return best_route

if __name__ == '__main__':
    N = int(input("N: "))
    cities = []
    for _ in range(N+1):
        cities.append(City())
    
    population_size = N*N if N > 4 else factorial(N)
    mutation_rate = 0.01
    elite_size = int(0.5*N*N)
    # if 40% of fitness values in a population are convergent then mark the population as not improving
    convergence_rate = 0.4
    
    best_route = get_best_route(cities, population_size, elite_size, mutation_rate, convergence_rate)
    print("Best route:")
    print(best_route)
    