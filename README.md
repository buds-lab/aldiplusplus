# ALDI++: Automatic and parameter-less discorddetection for daily load energy profiles
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)  ![Python Version](https://upload.wikimedia.org/wikipedia/commons/3/34/Blue_Python_3.6_Shield_Badge.svg) [![arXiv](https://img.shields.io/badge/arXiv-2203.06618-b31b1b.svg)](https://arxiv.org/abs/2203.06618)

Initial codebase: https://github.com/intelligent-environments-lab/ALDI

## Requirements

### Local

To run locally, you can execute the current environments:

```setup
conda env create --file env/environment_<OS>.yaml # replace OS with either `macos` or `ubuntu`
```

### AWS

For the forecasting portion of this project (training and prediction), we recommend using the following EC2 instance which was used in our experiments:
- Instance Type: `g4dn.4xlarge` (16 vCPUs, 64 GB RAM, and 600 GB disk)
- AMI: `Deep Learning AMI (Ubuntu 18.04)`
- Conda environment `tensorflow2_p36`

For the forecasting portion of this project, we recommend using the following EC2 instance which was used in our experiments:
- Instance Type: g4dn.4xlarge (16 vCPUs, 64 GB RAM, and 600 GB disk)
- AMI: Deep Learning AMI (Ubuntu 18.04)
- Conda environment `tensorflow2_p36`

## Data

We chose the following publicly available:

- [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2)

And specifically, the subset used for the [Great Energy Predictor III (GEPIII)](https://www.kaggle.com/c/ashrae-energy-prediction) machine learning competition.

Download the datasets from [`building-data-genome-project-2/data/meters/kaggle`](https://github.com/buds-lab/building-data-genome-project-2/tree/master/data/meters/kaggle) into `data/`.

The manually labeled outliers, from the top winning teams, are extracted from the following resources:
- [rank-1 winning team](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/tree/master/solutions/rank-1/input)
and are stored in `data/outliers`

## Discord detection using ALDI++

@TODO: Till can you fill out this part? which files to use and in what order? Thanks!

## Benchmarking models

- Statistical model (2-Standard deviation)
- [ALDI](https://doi.org/10.1016/j.enbuild.2020.109892)
- Variational Auto-encoder
- ALDI++ (our method)

## Evaluation
### Discord classification

Confusion matrices and ROC-AUC metrics are evaluated using the following notebooks:

`classification_<model>.ipynb`

where `<model>` is one of the benchmarked models: `2sd`, `vae`, `aldi`, `aldipp`

## Energy Forecasting

### Settings and parameters

To specify different settings and parameters pertinent to the data pre-processing, training, and evaluation, modify the files inside the `configs/` folder as a `yaml` file. The pipeline used for energy forecasting is based on the [Rank-1 team's solution](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/tree/master/solutions/rank-1).

It is assumed, however, that at least the following folder structure exists:

```
.
├── configs
│   ├── ..
├── data
│   ├── outliers
│   │   ├── ...
│   ├── preprocessed
│       ├── ...
...
```

### Training pipeline

Each `yaml` file inside `configs/` holds the configuration of different discord detection algorithms. Thus, in order to execute a strip-down version of the Rank-1 team's solution the following line needs to be executed:

```pipeline
./rank1-solution-simplified.sh configs/{your_config}.yaml
```

## Results

Our model achieves the following forecasting performance (`RMSLE`) and computation time (min) on the GEPIII dataset, the results of the original competition winning team, a simple statistical approach, a commonly used deep learning approch, and the original ALDI are shown too:

|   Discords labeled by   |  RMSLE | Computation time (min) |
| ----------------------- | ------ | ---------------------- |
| Kaggle winning team     |  2.841 |           480          |
| 2-Standard deviation    |  2.835 |             1          |
| ALDI                    |  2.834 |            40          |
| VAE                     |  2.829 |            32          |
| **ALDI++**              |  2.665 |             8          |

## Contributing

MIT License
