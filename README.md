# IVTransformer
This repository provides all experimental code for our paper:

**Transformers Handle Endogeneity in In-Context Linear Regression**.  

The code is built upon https://github.com/dtsip/in-context-learning and https://github.com/allenbai01/transformers-as-statisticians.
## Usage

### Create conda environment
```
conda env create -f environment.yml
```

### Activate environment
```
conda activate ivtransformer
```


### Pre-Training
Navigate to the `src/` directory and train the transformer model on IV regression task:
```
python train.py --config conf/gpt/iv_regression.yaml
```

### Track training progress with wandb
Edit `conf/wandb.yaml` with your wandb info.

### Runtime
The training process was conducted on a Windows 11 machine with the following specifications:

- GPU: NVIDIA GeForce RTX 4090
- CPU: Intel Core i9-14900KF
- Memory: 32 GB DDR5, 5600MHz

Total training time: Approximately 10 hours. 

The trained transformer model is saved under folder `\TrainedTF` (model not attached because of size limit). 

### Evaluation 
See `evaluation.ipynb` for details.

### Other simulation results
See `simulation.ipynb` for details

