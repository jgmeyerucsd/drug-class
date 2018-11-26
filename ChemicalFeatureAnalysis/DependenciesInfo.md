### Get jupyter notebooks running from the proper env

[Create a conda environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/):

```bash
conda create -n drug-class python=3.5.3 anaconda

conda install -c conda-forge rdkit
```

Install the right version of Python. I found a [link](https://github.com/rdkit/rdkit/issues/1402) to an issue that suggests you need a specific version of Python3 to get rdkit to work.

Install Python dependencies:
```bash
pip install --upgrade pip
pip install jupyter
pip install pubchempy
pip install seaborn
pip install sklearn
```

Had an issue with paths, so I had to call the ipython path manually when I was in the conda environment:
```bash
/Users/ijmiller2/anaconda3/envs/drug-class/bin/ipython
```

### To get out of the conda environment
source deactivate

### To get back in it
source activate drug-class
