# Part of 1st place solution (Simple LSTM) of Google-Brain-Ventilator competition

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description
Simple LSTM model used in the https://www.kaggle.com/c/ventilator-pressure-prediction competition.

## How to run
Install dependencies
```yaml
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
bash bash/setup_conda.sh

# install requirements
pip install -r requirements.txt
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```yaml
python run.py experiment=cloudy ++trainer.gpus=[0] ++datamodule.fold=0 
```

You can easily run all the folds by
```yaml
for fold in 0 1 2 3 4 5 6 7 8 9 10; do python run.py experiment=cloudy ++trainer.gpus=[0] ++datamodule.fold=$fold; done
```

You can easily get the oof and preds for a fold
```yaml
python run_inference.py experiment=cloudy ++trainer.gpus=[0] ++datamodule.fold=0
```

You can easily get all the oof and preds by fold using:
```yaml
for fold in 0 1 2 3 4 5 6 7 8 9 10; do python run_inference.py experiment=cloudy ++trainer.gpus=[0] ++datamodule.fold=$fold; done
```
oof and preds will be save inside the logs/experiments/cloudy/*fold_number*

<br>
