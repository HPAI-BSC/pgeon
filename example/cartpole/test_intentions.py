if __name__ == '__main__':
    from pgeon.policy_graph import PolicyGraph

    pg_pickle = PolicyGraph.from_pickle('./ppo-cartpole.pickle')

    print(f'Number of nodes:             {len(pg_pickle.nodes)}')
    print(f'Number of edges:             {len(pg_pickle.edges)}')
    print(f'Num. of stored trajectories: {len(pg_pickle._trajectories_of_last_fit)}')

    from pgeon.intention_introspector import IntentionIntrospector
    from pgeon.desire import Desire
    from pgeon.discretizer import Predicate
    from example.cartpole.discretizer import Position

    ii = IntentionIntrospector({Desire("keep_straight", 0, {Predicate(Position, [Position.MIDDLE])})})
    # %%
    print(ii.find_intentions(pg_pickle, 0.5))
