Log-revenue Mechanism: Improving DAG Throughput via Transaction Fee Design
=======

Environment
-----

In our DAG enviroment ([`DAGEnv`](./envs/DAGEnv.py)), episode step is 1, so `done` flag is always True, and $\hat{A}=r-V(s)$.

Config
-----

Base config is in [`./config/base.yaml`](./config/base.yaml). Part of the config parameters are from _TIPS: Transaction Inclusion Protocol With Signaling in DAG-Based Blockchain_

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

```bash
# Train PPO
python3 ./run.py --method PPO --mode train --cfg ./config/ppo.yaml
```
