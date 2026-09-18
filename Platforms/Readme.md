# Green platform

## Introduction

To retrieve the data as described in the “Model of edge resources” section of the article “Carbon-aware Scheduling on Multiple Edge Servers”, simply follow the setup instructions up to “Launch” section, then open the “GreenPlatform.ipynb” file and follow the instructions provided there.

## Setup

### Miniconda for Linux

```bash
cd /your/directory/
mkdir -p miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
chmod +x miniconda3/miniconda.sh
bash miniconda3/miniconda.sh -b -u -p miniconda3
rm -f miniconda3/miniconda.sh
```

### Virtual env

```bash
conda env create -f greenpfenv.yml
conda activate greenpfenv
```

### Launch

```bash
jupyter notebook
```

### See all env

```bash
conda env list
```

### Delete

```bash
conda env remove --name env_name
```
