# BaPPI-balanced learning for PPI prediction
This repository contains an official implementation of BaPPI and datasets used for evaluating PPI prediction model.
----
## Environments
- python3.11
- torch2.0.1
- keras2.13.1
- numpy1.24.3
- pandas2.0.3
----
### Usage
```
usage: PPIM [-h] [-m M] [-o O] [-i I] [-i1 I1] [-i2 I2] [-i3 I3] [-e E] [-b B] [-ln LN] [-L L]
            [-Loss LOSS] [-jk JK] [-ff FF] [-hl HL] [-sv SV] [-cuda CUDA] [-force FORCE]
            [-PSSM PSSM]

options:
  -h, --help    show this help message and exit
  -m M          mode, optinal value: read,bfs,dfs,rand,
  -o O
  -i I
  -i1 I1        sequence file
  -i2 I2        relation file
  -i3 I3        file path of test set indices (for read mode)
  -e E          epochs
  -b B          batch size
  -ln LN        graph layer num
  -L L          length for sequence padding
  -Loss LOSS    loss function
  -jk JK        use jump knowledege to fuse pair or not
  -ff FF        option for protein pair representaion
  -hl HL        hidden layer
  -sv SV        if save dataset path
  -cuda CUDA    if use cuda
  -force FORCE  if write to existed output file
  -PSSM PSSM    if use PSSM
```
### Sample command for training and testing
```
python3 main.py -m bfs -i SHS27K.txt  -L 512 -o SHS27K_bfs -ln 3 -e 100
