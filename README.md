Log-revenue Mechanism: Improving DAG Throughput via Transaction Fee Design
=======

Environment
-----

In our DAG enviroment ([`DAGEnv`](./envs/DAGEnv.py)), episode step is 1, so `done` flag is always True, and $\hat{A}=r-V(s)$.

The DAG environment uses the theorems 6 in _Inclusive Block Chain Protocols_ to calculate the transaction inclusion probability.

![DAGEnv](./assets/FC15.png)

Config
-----

Base config is in [`./config/base.yaml`](./config/base.yaml). Part of the config parameters are from _TIPS: Transaction Inclusion Protocol With Signaling in DAG-Based Blockchain_.

Data
-----

Data source: [https://gz.blockchair.com/bitcoin/transactions/](https://gz.blockchair.com/bitcoin/transactions/)

Get fee data:

```bash
python3 ./data/get_fee.py
```

Fee distribution:
mean: 7167.122512974324, std: 38358.98737426391
![fee distribution](./assets/fee_distribution.png)

Run
-----

* Train: train function approximator
* Test: test throughputs
* Eval: evaluate the equilibrium

```bash
# Train Agent
python3 ./run.py --method [DDPG/PPO] --mode train --cfg ./config/[ddpg/ppo].yaml
# Test throughputs
python3 ./run.py --method [DDPG/PPO] --mode test --cfg ./config/[ddpg/ppo].yaml
# Eval equilibrium
python3 ./run.py --method [DDPG/PPO] --mode eval --cfg ./config/[ddpg/ppo].yaml
```
