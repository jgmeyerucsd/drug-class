# Source Codes for Fingerprints-based  Methods

## Env Configuration

```
conda env create -f cpu_env.yml
source activate drug_class
pip install -U scikit-learn
```

## Random Forest

`python random_forest.py --weight_file=xxx.pt --number_of_class=[3/5/12] --index=[1-10]`