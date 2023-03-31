import math
import random

from dataclasses import dataclass, field

Chrome = list[int]


@dataclass
class GeneticAlgorithm:
    counter: int = 0
    max_gen: int = 200
    pop_size: int = 100
    prob_mut: float = 0.2
    prob_cross: float = 0.9
    npar: int = 2
    nbits: int = 8
    lchrome: int = npar * nbits
    vmax: list[int] = field(default_factory=lambda: [1, 1])
    vmin: list[int] = field(default_factory=lambda: [0, 0])
    max_bits: int = 2 ** nbits - 1

    def __post_init__(self) -> None:
        self.pop = self.init_pop()

    def random_chrome(self) -> Chrome:
        return [random.randint(0, 1) for _ in range(self.lchrome)]

    def init_pop(self) -> list[Chrome]:
        return [self.random_chrome() for _ in range(self.pop_size)]

    def decode(self, chrome: Chrome, i: int) -> float:
        start = i * self.nbits
        end = start + self.nbits
        return self.vmin[i] + (self.vmax[i] - self.vmin[i]) * int(''.join(map(str, chrome[start:end])),
                                                                  2) / self.max_bits

    def print_pop(self) -> None:
        for i, chrome in enumerate(self.pop):
            bit_str = ''.join(map(str, chrome))
            print(
                f'{i}: {bit_str} ({self.decode(chrome, 0)}, {self.decode(chrome, 1)}) fitness: {self.fitness(chrome)}')

    def fitness(self, chrome: Chrome) -> float:
        x = self.decode(chrome, 0)
        y = self.decode(chrome, 1)
        return 11 - (math.pi * x ** 3 * y - 0.1) ** 2

    def two_point_cross(self, chrome1: Chrome, chrome2: Chrome) -> tuple[Chrome, Chrome]:
        random_points = random.sample(range(1, self.lchrome - 1), 2)
        p1, p2 = min(random_points), max(random_points)
        d1 = chrome1[:p1] + chrome2[p1:p2] + chrome1[p2:]
        d2 = chrome2[:p1] + chrome1[p1:p2] + chrome2[p2:]
        return d1, d2

    def mutate_one_point(self, chrome: Chrome) -> Chrome:
        """
        Mutate one point in the chromosome

        :param chrome: chromosome
        :return: mutated chromosome
        """
        i = random.randint(0, self.lchrome)
        chrome[i] = 1 - chrome[i]
        return chrome

    def mutate(self, chrome: Chrome) -> Chrome:
        """
        Mutate the chromosome

        :param chrome: chromosome
        :return: mutated chromosome
        """
        for i in range(self.lchrome):
            if random.random() < self.prob_mut:
                chrome[i] = 1 - chrome[i]
        return chrome
