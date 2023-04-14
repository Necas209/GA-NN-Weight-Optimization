from genalg import GeneticAlgorithm


def main() -> None:
    ga = GeneticAlgorithm(
        prob_mut=0.1,
        max_gen=500,
    )
    """
    ga.print_pop()
    print(ga.decode(ga.pop[0], 0))
    p1, p2 = ga.pop[0], ga.pop[1]
    print(f'p1: {p1}')
    print(f'p2: {p2}')
    c1, c2 = ga.two_point_cross(p1, p2)
    print(f'c1: {c1}')
    print(f'c2: {c2}')
    chrome = ga.random_chrome()
    print(f'chrome: {chrome}')
    print(f'mutated: {ga.mutate_one_point(chrome)}')
    """
    ga.run()


if __name__ == '__main__':
    main()
