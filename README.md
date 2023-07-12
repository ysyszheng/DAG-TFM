DAG+TFM
=======

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
