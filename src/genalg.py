from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

ModelParams = np.ndarray[tuple[int], np.dtype[np.float64]]
Population = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def compute_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class GeneticAlgorithm:
    model: nn.Module
    population_size: int = 10
    mutation_rate: float = 0.05
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

    def crossover(self, parents1: Population, parents2: Population) -> Population:
        num_parents, num_params = parents1.shape
        crossover_points = np.random.randint(1, num_params, size=num_parents)
        mask = np.random.random(size=parents1.shape) < self.crossover_rate
        crossover_mask = np.arange(num_params) < crossover_points[:, np.newaxis]
        mask = np.logical_and(mask, crossover_mask)
        child1 = parents1 * np.logical_not(mask) + parents2 * mask
        child2 = parents2 * np.logical_not(mask) + parents1 * mask
        children = np.concatenate((child1, child2), axis=0)
        return children

    def mutate(self, children: Population) -> Population:
        mask_off = np.random.random(size=children.shape) < self.neuron_off_rate
        mask_mutate = np.random.random(size=children.shape) < self.mutation_rate
        children[mask_off] = 0.0
        children[mask_mutate] += np.random.normal(scale=0.1, size=np.sum(mask_mutate))
        return children

    def select_parents(self, population: Population) -> Population:
        scores = self.calculate_scores(population)
        normalized_scores = scores / np.sum(scores)
        parent_indices = np.random.choice(
            range(self.population_size),
            size=(self.population_size // 2, 2),
            replace=True,
            p=normalized_scores
        )
        parents = population[parent_indices]
        return parents

    def calculate_scores(self, population: Population) -> list[float]:
        return [self.fitness_fn(individual) for individual in population]

    def run(self) -> None:
        # Initialize population
        num_params = compute_num_params(self.model)
        population: Population = np.random.uniform(low=-1, high=1, size=(self.population_size, num_params))
        scores: list[float] = []
        # Run for num_generations
        for gen in range(self.num_generations):
            # Select parents
            parents = self.select_parents(population)
            # Vectorized crossover and mutation
            children = self.crossover(parents[:, 0], parents[:, 1])
            children_mutated = self.mutate(children)
            # Update population
            population = children_mutated
            # Preserve the best individual using elitism
            scores = self.calculate_scores(population)
            max_score = np.max(scores)
            # Update the best solution and score
            if max_score > self.best_score:
                self.best_score = max_score
                self.best_solution = population[np.argmax(scores)]
            # Preserve the best solution using elitism
            if self.elitism:
                worst_idx = np.argmin(scores)
                population[worst_idx] = self.best_solution
            # Save fitness score of best solution
            self.fitness_scores.append(self.best_score)
            # Print generation info
            if gen % self.on_generation_interval == 0:
                self.on_generation(gen, scores)
        self.on_generation(self.num_generations - 1, scores)

    def plot_fitness(self) -> None:
        """ Plots the fitness of the best solution over time """
        plt.plot(self.fitness_scores)
        plt.title("GA & PyTorch - Iteration vs. Fitness")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.show()

    def print_summary(self) -> None:
        """ Prints a summary of the best solution """
        print(f"Best solution fitness: {self.best_score}")
        print(f"Best solution: {self.best_solution}")
