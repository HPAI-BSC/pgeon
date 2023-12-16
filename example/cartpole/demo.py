import gymnasium as gym
from example.cartpole.discretizer import CartpoleDiscretizer, Position, Velocity, Angle, Action
from pgeon import Agent, Predicate
from ray.rllib.algorithms.algorithm import Algorithm
from pgeon import PolicyGraph


class CartpoleAgent(Agent):
    def __init__(self, path):
        self.agent = Algorithm.from_checkpoint(path)

    def act(self, state):
        return self.agent.compute_single_action(state)


if __name__ == '__main__':
    environment = gym.make('CartPole-v1')
    discretizer = CartpoleDiscretizer()

    agent = CartpoleAgent('checkpoints/PPO_CartPole-v1_1acbb_00000_0_2023-12-05_19-28-36/checkpoint_000000')

    pg = PolicyGraph(environment, discretizer)
    pg = pg.fit(agent, num_episodes=10, update=False)
    print(f'Number of nodes: {len(pg.nodes)}')
    print(f'Number of edges: {len(pg.edges)}')

    arbitrary_state = list(pg.nodes)[0]

    print(arbitrary_state)
    print(f'  Times visited: {pg.nodes[arbitrary_state]["frequency"]}')
    print(f'  p(s):          {pg.nodes[arbitrary_state]["probability"]:.3f}')

    arbitrary_edge = list(pg.edges)[0]

    print(f'From:    {arbitrary_edge[0]}')
    print(f'Action:  {arbitrary_edge[2]}')
    print(f'To:      {arbitrary_edge[1]}')
    print(f'  Times visited:      {pg[arbitrary_edge[0]][arbitrary_edge[1]][arbitrary_edge[2]]["frequency"]}')
    print(f'  p(s_to,a | s_from): {pg[arbitrary_edge[0]][arbitrary_edge[1]][arbitrary_edge[2]]["probability"]:.3f}')

    possible_actions = pg.question1((
        Predicate(Position, [Position(Position.MIDDLE)]),
        Predicate(Velocity, [Velocity(Velocity.RIGHT)]),
        Predicate(Angle, [Angle(Angle.STANDING)])))

    print('I will take one of these actions:')
    for action, prob in possible_actions:
        print('\t->', action.name, '\tProb:', round(prob * 100, 2), '%')

    best_states = pg.question2(Action.LEFT.value)
    print(f'I will perform action {0} in these states:')
    print('\n'.join([str(state) for state in best_states]))

    print(f'Supposing I was in the middle, moving right, with the pole standing upright, '
          f'if I did not choose to move left was due to...')
    counterfactuals = pg.question3((
        Predicate(Position, [Position(Position.MIDDLE)]),
        Predicate(Velocity, [Velocity(Velocity.RIGHT)]),
        Predicate(Angle, [Angle(Angle.STANDING)])), Action.LEFT)
    for ct in counterfactuals:
        print(f'...{" and ".join([str(i[0]) + " -> " + str(i[1]) for i in ct.values()])}')

