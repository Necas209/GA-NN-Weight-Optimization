import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

Weights = np.ndarray[float]
Pair = tuple[Weights, Weights]


def avg_fitness(fitness_scores: list[float]) -> float:
    return np.mean(fitness_scores).item()


def best_fitness(fitness_scores: list[float]) -> float:
    return np.max(fitness_scores)


def worst_fitness(fitness_scores: list[float]) -> float:
    return np.min(fitness_scores)


@dataclass
class GeneticAlgorithm:
    model: nn.Module
    train_loader: data.DataLoader
    test_loader: data.DataLoader
    population: np.ndarray[Weights] = None
    population_size: int = 10
    mutation_rate: float = 0.01
    crossover_rate: float = 0.95
    num_generations: int = 100
    best_solutions: list[Weights] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.population = np.random.uniform(
            low=-1,
            high=1,
            size=(self.population_size, self.num_weights)
        )

    def load_weight(self, weights: Weights) -> None:
        model_params = self.model.state_dict()
        model_params['0.weight'] = torch.FloatTensor(weights[:40].reshape(10, 4))
        model_params['0.bias'] = torch.FloatTensor(weights[40:50])
        model_params['2.weight'] = torch.FloatTensor(weights[50:80].reshape(3, 10))
        model_params['2.bias'] = torch.FloatTensor(weights[80:])
        self.model.load_state_dict(model_params)

    @property
    def num_weights(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def on_generation(self, generation: int) -> None:
        scores = self.fitness_scores
        self.best_solutions.append(self.best_solution(scores))
        print(
            f"Generation: {generation} "
            f"Best fitness: {best_fitness(scores)} "
            f"Average fitness: {avg_fitness(scores)} "
            f"Worst fitness: {worst_fitness(scores)}"
        )

    def fitness_function(self, weights: Weights) -> float:
        self.load_weight(weights)
        accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                labels: torch.Tensor
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                accuracy += (predicted == labels).sum().item() / len(labels)
        accuracy /= len(self.test_loader)
        return accuracy

    def crossover(self, parent1: Weights, parent2: Weights) -> Pair:
        crossover_point = random.randint(1, len(parent1) - 1)
        if random.random() > self.crossover_rate:
            return parent1, parent2
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, individual: Weights) -> Weights:
        mutated_individual = individual.copy()
        for i in range(len(mutated_individual)):
            if random.random() < self.mutation_rate:
                mutated_individual[i] += np.random.normal(scale=0.1)
        return mutated_individual

    def select_parents(self, population: np.ndarray[Weights], fitness_scores: list[float]) -> list[Pair]:
        parents = []
        for _ in range(int(self.population_size / 2)):
            parent1 = population[random.choices(range(self.population_size), weights=fitness_scores)[0]]
            parent2 = population[random.choices(range(self.population_size), weights=fitness_scores)[0]]
            parents.append((parent1, parent2))
        return parents

    @property
    def fitness_scores(self) -> list[float]:
        return [self.fitness_function(individual) for individual in self.population]

    def best_solution(self, fitness_scores: list[float]) -> Weights:
        return self.population[np.argmax(fitness_scores)]

    def run(self) -> Weights:
        for i in range(self.num_generations):
            parents = self.select_parents(self.population, self.fitness_scores)

            offspring = []
            for parent1, parent2 in parents:
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.append(child1)
                offspring.append(child2)
            self.population = np.array(offspring)
            self.on_generation(i)

        best_individual = self.population[np.argmax(self.fitness_scores)]
        return best_individual
