Log-revenue Mechanism: Improving DAG Throughput via Transaction Fee Design
=======

Environment
-----

In our DAG enviroment ([`DAGEnv`](./envs/DAGEnv.py)), episode step is 1, so `done` flag is always True.

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
![fee distribution](./assets/fee_distribution.png)

Run
-----

```bash
python3 ./run.py --method PPO --mode train --cfg ./config/ppo.yaml
```
