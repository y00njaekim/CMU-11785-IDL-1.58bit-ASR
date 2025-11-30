## Steps to run - 

first prepare data - 

1. download data using utils/download_data.sh
2. train tokenizer - python src/data/tokenizer.py
3. compute cmvn - python main.py

now, train -

1. Set up WandB API key (required for logging):
   - Create a `.env` file in the project root with:
     ```
     WANDB_API_KEY=your-wandb-api-key-here
     ```
   - Get your API key from: https://wandb.ai/authorize
   - The `.env` file is already in `.gitignore`, so it won't be committed

2. Run training:
   ```bash
   python -m onebit_asr.train
   ```


## BUGS and TODO - 

1. training only supports v small batch size (less than 8).