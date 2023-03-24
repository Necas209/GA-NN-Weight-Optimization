import random
from dataclasses import dataclass, field


@dataclass
class GeneticAlgorithm:
    counter: int = 0
    max_gen: int = 200
    pop_size: int = 100
    pm: float = 0.2
    pc: float = 0.9
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
        return self.vmin[i] + (self.vmax[i] - self.vmin[i]) * sum(chrome[start:end]) / self.max_bits


def main() -> None:
    ga = GeneticAlgorithm(pop_size=20)
    print(ga.random_chrome())


if __name__ == '__main__':
    main()
