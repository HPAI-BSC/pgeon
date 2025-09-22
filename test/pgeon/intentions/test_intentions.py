import csv
import unittest
from pathlib import Path

import numpy as np

from test.pgeon.intentions.predicates import (
    Action2Nearest,
    Agent,
    Direction,
    Held,
    Object,
    PotState,
    PredicateName,
)

import pandas as pd

from pgeon.desire import Desire
from pgeon.discretizer import (
    Discretizer,
    Predicate,
    PredicateBasedState,
    State,
    Transition,
)
from pgeon.intention_aware_policy_approximator import (
    IntentionalStateMetadata,
    IntentionAwarePolicyApproximator,
)
from pgeon.policy_representation import GraphRepresentation

NODES_PATH = Path(__file__).parent / "pg_simple_14_nodes.csv"
EDGES_PATH = Path(__file__).parent / "pg_simple_14_edges.csv"
INTENTIONS_PATH = Path(__file__).parent / "intentions_simple_D14_C0.5.csv"

pred_classes = [Held, PotState, Action2Nearest, Direction, Agent, Object, PredicateName]
pred_enum_values = {
    value.name: class_
    for class_ in pred_classes
    for value in class_.__members__.values()
}


class OvercookedDiscretizer(Discretizer):
    def discretize(self, state): ...

    def state_to_str(self, state: State): ...
    def nearest_state(self, state: State): ...
    def all_actions(self): ...


def str_to_state(state_str: str):
    preds: list[Predicate] = []
    for pred_str in state_str.split("+"):
        pred_name_str = pred_str.split("(")[0]
        pred_name = pred_enum_values[pred_name_str][pred_name_str]
        pred_args = []
        for pred_arg_str in pred_str.split("(")[1].split(")")[0].split(";"):
            pred_arg = pred_enum_values[pred_arg_str][pred_arg_str]
            pred_args.append(pred_arg)
        preds.append(Predicate(pred_name, tuple(pred_args)))
    return PredicateBasedState(preds)


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

    node_ids_to_values: dict[int, State] = {}
    with open(nodes_path, "r+") as f:
        csv_r = csv.reader(f)
        next(csv_r)

        for row in csv_r:
            state_id = int(row[0])
            value = str_to_state(row[1])
            prob = float(row[2])

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
    return representation, node_ids_to_values


class TestIntentions(unittest.TestCase):
    def test_load_nodes_and_edges(self):
        pr, _ = load_nodes_and_edges(NODES_PATH, EDGES_PATH)
        self.assertEqual(len(list(pr.states)), 2918)
        self.assertEqual(len(list(pr.transitions)), 13904)

    def test_propagate_intentions(self):
        pr, ids = load_nodes_and_edges(NODES_PATH, EDGES_PATH)
        pa = IntentionAwarePolicyApproximator(None, pr, None, None)

        desire_to_service = Desire(
            "desire_to_service",
            5,
            PredicateBasedState(
                [
                    Predicate(PredicateName.HELD_PLAYER, (Held.SOUP,)),
                    Predicate(
                        PredicateName.ACTION2NEAREST,
                        (Object.SERVICE, Action2Nearest.INTERACT),
                    ),
                ]
            ),
        )

        # clause = {'HELD_PLAYER(ONION)', 'ACTION2NEAREST(POT0;INTERACT)', 'POT_STATE(POT0;PREPARING)'}
        # action = action_name_to_idx['Interact']
        # desire_to_cook0 = Desire("desire_to_cook0", action, clause)

        # clause = {'HELD_PLAYER(ONION)', 'ACTION2NEAREST(POT0;INTERACT)', 'POT_STATE(POT0;NOT_STARTED)'}
        # action = action_name_to_idx['Interact']
        # desire_to_start_cooking0 = Desire("desire_to_start_cooking0", action, clause)

        # desires = [desire_to_service, desire_to_cook0, desire_to_start_cooking0]

        pa.register_desire(desire_to_service)

        golden_truth = (
            pd.read_csv(INTENTIONS_PATH)
                .set_index("node_id")
                .loc[:, ["desire_to_service"]]
        )

        for state_id, series in golden_truth.iterrows():
            truth_datum = series["desire_to_service"]
            if not np.isnan(truth_datum):
                self.assertAlmostEqual(truth_datum, pr.states[ids[state_id]].metadata.intention[desire_to_service])


if __name__ == "__main__":
    unittest.main()
