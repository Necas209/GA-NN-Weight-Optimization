import random
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ModelParams = np.ndarray[float]
Population = list[ModelParams] | np.ndarray[ModelParams]


@dataclass
class GeneticAlgorithm:
    model: nn.Module
    train_set: tuple[torch.Tensor, torch.Tensor]
    population_size: int = 10
    mutation_rate: float = 0.01
    crossover_rate: float = 0.95
    num_generations: int = 100
    on_generation_interval: int = 10
    best_solutions: list[ModelParams] = field(default_factory=list)
    fitness_fn: Callable[[nn.Module, ModelParams], float] = None
    on_generation: Callable[[int, list[float]], None] = None

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def crossover(self, parent1: ModelParams, parent2: ModelParams) -> tuple[ModelParams, ModelParams]:
        crossover_point = random.randint(1, len(parent1) - 1)
        if random.random() > self.crossover_rate:
            return parent1, parent2
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, individual: ModelParams) -> ModelParams:
        mutated_individual = individual.copy()
        for i in range(len(mutated_individual)):
            if random.random() < self.mutation_rate:
                mutated_individual[i] += np.random.normal(scale=0.1)
        return mutated_individual

    def select_parents(self, population: Population) -> list[tuple[ModelParams, ModelParams]]:
        scores = self.fitness_scores(population)
        parents = []
        for _ in range(int(self.population_size / 2)):
            parent1 = population[random.choices(range(self.population_size), weights=scores)[0]]
            parent2 = population[random.choices(range(self.population_size), weights=scores)[0]]
            parents.append((parent1, parent2))
        return parents

    def fitness_scores(self, population: Population) -> list[float]:
        return [self.fitness_fn(self.model, individual) for individual in population]

    def run(self) -> None:
        # Initialize population
        population = np.random.uniform(low=-1, high=1, size=(self.population_size, self.num_params))
        # Run for num_generations
        for gen in range(self.num_generations):
            # Select parents
            parents = self.select_parents(population)
            # Generate offspring
            offspring = []
            for parent1, parent2 in parents:
                # Crossover and mutate
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.append(child1)
                offspring.append(child2)
            # Select new population
            population = np.array(offspring)
            # Save best individual
            scores = self.fitness_scores(population)
            best_individual = population[np.argmax(scores)]
            self.best_solutions.append(best_individual)
            # Print generation info
            if gen % self.on_generation_interval == 0:
                self.on_generation(gen, scores)

    @property
    def best_solution(self) -> ModelParams:
        """ Returns the best overall solution """
        return max(self.best_solutions, key=partial(self.fitness_fn, self.model))

    def plot_fitness(self) -> None:
        """ Plots the fitness of the best solution over time """
        scores = self.fitness_scores(self.best_solutions)
        plt.plot(scores)
        plt.title("GA & PyTorch - Iteration vs. Fitness")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.show()
