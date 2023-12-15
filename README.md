# pgeon

> Tormos, A., Abalos, V., Gnatyshak, D., & Alvarez-Napagao, S. (2023, October).  [Policy graphs in action: explaining single- and multi-agent behaviour using predicates](https://openreview.net/forum?id=QPqL9xsYOf). In _XAI in Action: Past, Present, and Future Applications_.

**_pgeon_** is a Python package that produces explanations for opaque agents using **Policy Graphs** (PGs).

A Policy Graph is a means to obtain a representation of the behavior of an opaque agent, in the form of a directed graph. Discrete states are mapped to nodes and actions to edges.

## Getting started

1. Download the `pgeon/` folder and move it into the root directory of your project.
2. Install **pgeon**'s requirements with `pip install -r PATH_TO_PGEON_REQUIREMENTS`.


## Example usage

### Generating a Policy Graph

Given a Gymnasium `environment` and a `discretizer`, you can generate a PG to describe an opaque agent's behavior with `fit()`.
```python
from pgeon import PolicyGraph

pg = PolicyGraph(environment, discretizer)
pg = pg.fit(agent, num_episodes=1000)
```

### Creating and using a PG-based policy

There exist two PG-based policies. You can select one or the other with `PGBasedPolicyMode`.

```python
from pgeon import PGBasedPolicy, PGBasedPolicyMode

greedy_policy = PGBasedPolicy(pg, mode=PGBasedPolicyMode.GREEDY)
stochastic_policy = PGBasedPolicy(pg, mode=PGBasedPolicyMode.STOCHASTIC)

# Passing the policy an observation to get an action
obs, _ = environment.reset()
action = greedy_policy.act(obs)
```

### More examples

You can check [`examples/cartpole/demo.ipynb`](https://github.com/HPAI-BSC/pgeon/blob/main/example/cartpole/demo.ipynb) for a complete breakdown of **pgeon**'s features.

To run the notebook yourself:

1. Download the entire repository.
2. Install **pgeon**'s requirements with `pip install -r requirements.txt`.
3. Install an extra dependency, rllib, with `pip install "ray[rllib]"`.
4. Open and execute `examples/cartpole/demo.ipynb`.

## Citation

If you use the **pgeon** library, please cite:

```
@inproceedings{tormos2023policy,
  title={Policy graphs in action: explaining single-and multi-agent behaviour using predicates},
  author={Tormos, Adrian and Abalos, Victor and Gnatyshak, Dmitry and Alvarez-Napagao, Sergio},
  booktitle={XAI in Action: Past, Present, and Future Applications},
  year={2023},
  url={https://openreview.net/forum?id=QPqL9xsYOf}
}
```
