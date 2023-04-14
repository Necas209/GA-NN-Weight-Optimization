import random

from dataclasses import dataclass, field

Chrome = list[int]


@dataclass
class GeneticAlgorithm:
    pop: list[Chrome] = field(default_factory=list)
    counter: int = 0
    max_gen: int = 200
    pop_size: int = 100
    prob_mut: float = 0.2
    prob_cross: float = 0.9
    npar: int = 2
    nbits: int = 8
    lchrome: int = npar * nbits
    vmax: list[int] = field(default_factory=lambda: [5, 5])
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
                f'{i}: {bit_str} '
                f'({self.decode(chrome, 0)}, {self.decode(chrome, 1)}) '
                f'fitness: {self.fitness(chrome)}'
            )

    def fitness(self, chrome: Chrome) -> float:
        x = self.decode(chrome, 0)
        y = self.decode(chrome, 1)
        return (x - 3) ** 2 + (y - 2) ** 2

    def two_point_cross(self, chrome1: Chrome, chrome2: Chrome) -> tuple[Chrome, Chrome]:
        random_points = random.sample(range(1, self.lchrome), 2)
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

    def select_chrome(self) -> Chrome:
        """
        Select a chromosome from the population

        :return: selected chromosome
        """
        return random.choice(self.pop)

    def tournament(self, n: int = 2) -> Chrome:
        """
        Select a chromosome from the population using tournament selection

        :param n: number of participants. Default is 2
        :return: selected chromosome
        """
        if n > len(self.pop):
            n = len(self.pop)
        pop = random.sample(self.pop, n)
        return min(pop, key=self.fitness)

    @property
    def best_fitness(self) -> float:
        """
        Get the best fitness of the population

        :return: best fitness
        """
        return min(map(self.fitness, self.pop))

    @property
    def worst_fitness(self) -> float:
        """
        Get the worst fitness of the population

        :return: worst fitness
        """
        return max(map(self.fitness, self.pop))

    @property
    def avg_fitness(self) -> float:
        """
        Get the average fitness of the population

        :return: average fitness
        """
        return sum(map(self.fitness, self.pop)) / len(self.pop)

    def run(self, print_freq: int = 10) -> None:
        """
        Run the genetic algorithm
        """
        while self.counter < self.max_gen:
            new_pop = []
            for _ in range(self.pop_size):
                p1 = self.tournament()
                p2 = self.tournament()
                if random.random() < self.prob_cross:
                    c1, c2 = self.two_point_cross(p1, p2)
                else:
                    c1, c2 = p1, p2
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_pop.append(c1)
                new_pop.append(c2)
            self.pop = new_pop
            self.counter += 1
            if self.counter % print_freq == 0:
                print(
                    f'Gen: {self.counter: >5} '
                    f'Best: {self.best_fitness: >8.5f} '
                    f'Worst: {self.worst_fitness: >8.5f} '
                    f'Avg: {self.avg_fitness: >8.5f}'
                )
