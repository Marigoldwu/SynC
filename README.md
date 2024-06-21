An official code for paper "Synergistic Deep Graph Clustering Network".

## Reproducibility

To reproduce the results in our paper locally, you should follow these steps:

Step :one: Download the repository to `SynC`.

```bash
git clone https://github.com/Marigoldwu/SynC SynC
```

Step :two: Create a python virtual environment (`conda`) and install the dependencies:

```bash
conda create --name sync python=3.8
conda activate sync
pip install -r requirements.txt
```

PyTorch is required (You can choose a suitable version according to your device):

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

Step :three: Prepare datasets (unzip the .rar file). All the datasets can be fetched from Liu's repository [[Link](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering?tab=readme-ov-file#benchmark-datasets)]. The data files are organized as follows:

```
SynC/
├── dataset/
│   ├── acm/
│   │   ├── acm_adj.npy    	# Dense adjacency matrix.
│   │   ├── acm_feat.npy   	# Feature matrix.
│   │   └── acm_label.npy  	# Ground-truth labels.
│   └── dataset_info.py    	# Dataset information, e.g. number of clusters, number of nodes...
```

Step :four: Train with our provided pre-training weights. 

```bash
cd SynC
python main.py -M SYNC -D acm -LS 10 -S 325
# Use the following command to view optional configuration information.
python main.py --help
```

If you want to train SynC on other datasets (provided in `dataset_info.py `, others need to manually add dataset information in this file), you can pre-train with the following command:
```bash
python main.py -P -M pretrain_tigae_for_sync -D acm -LS 1 -S 325
```

## Results

### The results in our paper.

|           | ACC        | NMI        | ARI        | F1         |
| --------- | ---------- | ---------- | ---------- | ---------- |
| ACM       | 92.73±0.04 | 73.58±0.22 | 79.58±0.11 | 92.74±0.04 |
| DBLP      | 83.48±0.13 | 55.11±0.24 | 61.70±0.27 | 82.90±0.17 |
| CITE      | 71.77±0.27 | 46.37±0.42 | 48.09±0.45 | 65.72±0.36 |
| CORA      | 78.58±0.38 | 58.13±0.52 | 57.90±1.06 | 77.65±0.30 |
| AMAP      | 82.48±0.04 | 69.70±0.23 | 65.02±0.11 | 80.69±0.11 |
| UAT       | 57.33±0.13 | 28.58±0.24 | 26.60±0.17 | 57.34±0.23 |
| Wisconsin | 59.64±1.02 | 32.79±1.42 | 26.86±1.30 | 38.19±1.30 |
| Texas     | 64.37±0.80 | 27.61±1.00 | 32.65±1.91 | 39.49±1.53 |

### The results reproduced by code ocean.

The code ocean capsule link: [Under review, will be released soon]()

The results on eight datasets recorded in the `console output.txt`.

> To reproduce our results, you can click the `Reproducible Run` button.
>
> Please note that the environment is slightly different from what is described in the article.  Even so, the results in our paper are easy to reproduce.

The code ocean Dockerfile:

```dockerfile
# hash:sha256:fe8085911e8d9a8c9a97a82e9bff996f985a0243f467d1578c08a7a47bfa0654
FROM registry.codeocean.com/codeocean/pytorch:2.1.0-cuda11.8.0-mambaforge23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        unrar=1:6.1.5-1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    matplotlib==3.7.5 \
    munkres==1.1.4 \
    numpy==1.24.1 \
    scikit-learn==1.3.2 \
    scipy==1.9.1
```

