# pgeon / pgeon-xai

<div align="center">
  
[![Paper](https://img.shields.io/badge/Paper-10.5555/3635637.3663299-f6c628.svg)](https://dl.acm.org/doi/10.5555/3635637.3663299)
<br/>
[![Website](https://img.shields.io/badge/Website-HPAI-8A2BE2.svg)](https://hpai.bsc.es)
[![GitHub](https://img.shields.io/badge/GitHub-HPAI--BSC-%23121011.svg?logo=github&logoColor=white.svg)](https://github.com/HPAI-BSC)
![GitHub Repo stars](https://img.shields.io/github/stars/HPAI-BSC/pgeon)
![GitHub followers](https://img.shields.io/github/followers/HPAI-BSC)
<br/>
[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HPAI--BSC-ffc107?color=ffc107&logoColor=white.svg)](https://huggingface.co/HPAI-BSC)
[![LinkedIn](https://img.shields.io/badge/Linkedin-HPAI--BSC-blue.svg)](https://www.linkedin.com/company/hpai)
[![BlueSky](https://img.shields.io/badge/Bluesky-HPAI-0285FF?logo=bluesky&logoColor=fff.svg)](https://bsky.app/profile/hpai.bsky.social)
[![LinkTree](https://img.shields.io/badge/Linktree-HPAI-43E55E?style=flat&logo=linktree&logoColor=white.svg)](https://linktr.ee/hpai_bsc)


</div>

**_pgeon_** (**_pgeon-xai_**) is a Python package that produces explanations for opaque agents using **Policy Graphs** (PGs).

A Policy Graph is a means to obtain a representation of the behavior of an opaque agent, in the form of a directed graph. Discrete states are mapped to nodes and actions to edges.

## Getting started

* Install **pgeon** with pip: `pip install pgeon-xai`

or:

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

> Tormos, A., Gimenez-Abalos, V., Vázquez-Salceda, J., & Alvarez-Napagao, S. (2024, May). [pgeon applied to Overcooked-AI to explain agents' behaviour](https://dl.acm.org/doi/10.5555/3635637.3663299). In _Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems_ (pp. 2821-2823).

```
@inproceedings{tormos2024pgeon,
  title={pgeon applied to {Overcooked-AI} to explain agents' behaviour},
  author={Tormos, Adrian and Gimenez-Abalos, Victor and V{\'a}zquez-Salceda, Javier and Alvarez-Napagao, Sergio},
  booktitle={Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems},
  pages={2821--2823},
  year={2024}
}
```
