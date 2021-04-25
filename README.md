# Explainable IQA on Chest X-Rays

Accepted in MIDL'21: https://openreview.net/forum?id=ln797A8lAb0.

This repository contains the necessary environment to perform explainable medical image quality analysis.


# Preparatory Steps

1. Create a new environment named eiqa and activate it by:

    `conda create --name eiqa python=3.7`

    `conda activate eiqa`

2. Install the necessary packages by entering:

    `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`

3. Create a new folder named `images` with a following folder tree structure for the object-CXR dataset. The dataset can be found at [here](https://academictorrents.com/details/fdc91f11d7010f7259a05403fc9d00079a09f5d5/tech).

```
images
├── object-CXR_test
│   └── test.csv
│   └── images
|   │   └── 09001.jpg
|   │   └── 09002.jpg
            .
            .
├── object-CXR_val
│   └── dev.csv
│   └── images
|   │   └── 08001.jpg
|   │   └── 08002.jpg
            .
            .
```

# Usage

The results of Table 2 in our paper can be reproduced by typing in `./reproduce_table2.sh`. It will not only show the Pointing Game accuracies, but also record the attribution maps corresponding each image in each split.

# Citation

```
@inproceedings{Ozer2021,
    title={Explainable Image Quality Analysis of Chest X-Rays},
    author={Ozer, Caner and Oksuz Ilkay}
    booktitle={Medical Imaging with Deep Learning}
    year={2021}
    url={https://openreview.net/forum?id=ln797A8lAb0}
}
```