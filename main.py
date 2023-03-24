import math
import random

from dataclasses import dataclass, field


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

    def random_chrome(self) -> list[int]:
        return [random.randint(0, 1) for _ in range(self.lchrome)]

    def init_pop(self) -> list[list[int]]:
        return [self.random_chrome() for _ in range(self.pop_size)]

    def decode(self, chrome: list[int], i: int) -> float:
        start = i * self.nbits
        end = start + self.nbits
        return self.vmin[i] + (self.vmax[i] - self.vmin[i]) * int(''.join(map(str, chrome[start:end])),
                                                                  2) / self.max_bits

    def print_pop(self) -> None:
        for i, chrome in enumerate(self.pop):
            bit_str = ''.join(map(str, chrome))
            print(
                f'{i}: {bit_str} ({self.decode(chrome, 0)}, {self.decode(chrome, 1)}) fitness: {self.fitness(chrome)}')

    def fitness(self, chrome: list[int]) -> float:
        x = self.decode(chrome, 0)
        y = self.decode(chrome, 1)
        return 11 - (math.pi * x ** 3 * y - 0.1) ** 2


def main() -> None:
    ga = GeneticAlgorithm(pop_size=20)
    print(ga.random_chrome())
    ga.print_pop()
    print(ga.decode(ga.pop[0], 0))


if __name__ == '__main__':
    main()
