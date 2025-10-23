## Steps to run - 

first prepare data - 

1. download data using utils/download_data.sh
2. train tokenizer - python src/data/tokenizer.py
3. compute cmvn - python main.py

now, train -

python -m onebit_asr.train


## BUGS and TODO - 

1. training only supports v small batch size (less than 8). this is a bug in either dataloader or conformer implementation

2. loss/gradients go to NaNs after a while