import csv
import unittest
from pathlib import Path

from pgeon.discretizer import Transition
from pgeon.intention_aware_policy_approximator import (
    IntentionalStateMetadata,
    IntentionAwarePolicyApproximator,
)
from pgeon.policy_representation import GraphRepresentation

NODES_PATH = Path(__file__).parent / "pg_simple_14_nodes.csv"
EDGES_PATH = Path(__file__).parent / "pg_simple_14_edges.csv"


def load_nodes_and_edges(nodes_path: Path, edges_path: Path):

    if not nodes_path.suffix == ".csv":
        raise ValueError(f"Nodes file must have a .csv extension, got {nodes_path}")
    if not edges_path.suffix == ".csv":
        raise ValueError(f"Edges file must have a .csv extension, got {edges_path}")

    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes file {nodes_path} does not exist")
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file {edges_path} does not exist")

    representation = GraphRepresentation("networkx", IntentionalStateMetadata)

    node_ids_to_values = {}
    with open(nodes_path, "r+") as f:
        csv_r = csv.reader(f)
        next(csv_r)

        for row in csv_r:
            state_id = int(row[0])
            value = str(row[1])
            prob = float(row[2])
            freq = None

            representation.graph.add_node(
                value,
                probability=prob,
            )
            node_ids_to_values[state_id] = value

    with open(edges_path, "r+") as f:
        csv_r = csv.reader(f)
        next(csv_r)

        for row in csv_r:
            node_from_id = int(row[0])
            node_to_id = int(row[1])
            action = int(row[2])
            prob = float(row[3])

            node_from = node_ids_to_values[int(node_from_id)]
            node_to = node_ids_to_values[int(node_to_id)]

            representation._add_transition(
                node_from,
                node_to,
                Transition(action=action, probability=prob),
            )
    return representation


class TestIntentions(unittest.TestCase):
    def test_load_nodes_and_edges(self):
        pr = load_nodes_and_edges(NODES_PATH, EDGES_PATH)
        self.assertEqual(len(list(pr.states)), 2918)
        self.assertEqual(len(list(pr.transitions)), 13904)

    def test_propagate_intentions(self):
        pr = load_nodes_and_edges(NODES_PATH, EDGES_PATH)
        pga = IntentionAwarePolicyApproximator(None, pr, None, None)
        # pga.register_desire(Desire("reach_one", 0, PredicateBasedState([Predicate(DummyState.ONE)])))


if __name__ == "__main__":
    unittest.main()
