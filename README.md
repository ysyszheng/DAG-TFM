Log-revenue Mechanism: Improving DAG Throughput via Transaction Fee Design
=======

Environment
-----

In our DAG enviroment ([`DAGEnv`](./envs/DAGEnv.py)), episode step is 1, so `done` flag is always True.

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
# train
python3 ./run.py --mode train --fn <data save to file>
# test
python3 ./run.py --mode test --fn <data save to file>
# verify
python3 ./run.py --mode verify --fn <data save to file>
# plot transaction fee - private value
python3 python3 ./utils/scatter.py --file <file name> --step <step number>
```
