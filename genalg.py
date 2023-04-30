from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ModelParams = np.ndarray[float]
Population = np.ndarray[ModelParams]


@dataclass
class GeneticAlgorithm:
    model: nn.Module
    population_size: int = 10
    mutation_rate: float = 0.01
    neuron_off_rate: float = 1e-3
    crossover_rate: float = 0.95
    elitism: bool = True
    num_generations: int = 100
    on_generation_interval: int = 10
    best_score: float = 0.0
    best_solution: ModelParams = None
    fitness_scores: list[float] = field(default_factory=list)
    fitness_fn: Callable[[ModelParams], float] = None
    on_generation: Callable[[int, list[float]], None] = None

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def crossover(self, parents1: Population, parents2: Population) -> tuple[Population, Population]:
        num_parents, num_params = parents1.shape
        crossover_points = np.random.randint(1, num_params, size=num_parents)
        mask = np.random.random(size=parents1.shape) < self.crossover_rate
        crossover_mask = np.arange(num_params) < crossover_points[:, np.newaxis]
        mask = np.logical_and(mask, crossover_mask)
        child1 = parents1 * np.logical_not(mask) + parents2 * mask
        child2 = parents2 * np.logical_not(mask) + parents1 * mask
        return child1, child2

    def mutate(self, children: Population) -> Population:
        mask_off = np.random.random(size=children.shape) < self.neuron_off_rate
        mask_mutate = np.random.random(size=children.shape) < self.mutation_rate
        children[mask_off] = 0.0
        children[mask_mutate] += np.random.normal(scale=0.1, size=np.sum(mask_mutate))
        return children

    def select_parents(self, population: Population) -> tuple[Population, Population]:
        scores = self.calculate_scores(population)
        normalized_scores = scores / np.sum(scores)
        parent_indices = np.random.choice(
            range(self.population_size),
            size=(self.population_size // 2, 2),
            replace=True,
            p=normalized_scores
        )
        parents1 = population[parent_indices[:, 0]]
        parents2 = population[parent_indices[:, 1]]
        return parents1, parents2

    def calculate_scores(self, population: Population) -> list[float]:
        return [self.fitness_fn(individual) for individual in population]

    def run(self) -> None:
        # Initialize population
        population: Population = np.random.uniform(low=-1, high=1, size=(self.population_size, self.num_params))
        # Initialize best solution
        self.best_solution = np.zeros(self.num_params)
        # Run for num_generations
        for gen in range(self.num_generations):
            # Select parents
            parents1, parents2 = self.select_parents(population)
            # Vectorized crossover and mutation
            child1, child2 = self.crossover(parents1, parents2)
            child1_mutated = self.mutate(child1)
            child2_mutated = self.mutate(child2)
            # Update population
            population[::2] = child1_mutated
            population[1::2] = child2_mutated
            # Preserve the best individual using elitism
            scores = self.calculate_scores(population)
            # Update the best solution and score
            if gen == 0:
                self.best_score = np.max(scores)
                self.best_solution = population[np.argmax(scores)]
            else:
                self.best_score = max(self.best_score, np.max(scores))
            # Preserve the best solution using elitism
            if self.elitism and gen > 0:
                worst_idx = np.argmin(scores)
                population[worst_idx] = self.best_solution
            # Save fitness score of best solution
            self.fitness_scores.append(self.best_score)
            # Print generation info
            if gen % self.on_generation_interval == 0 or gen == self.num_generations - 1:
                self.on_generation(gen, scores)

    def plot_fitness(self) -> None:
        """ Plots the fitness of the best solution over time """
        plt.plot(self.fitness_scores)
        plt.title("GA & PyTorch - Iteration vs. Fitness")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.show()
