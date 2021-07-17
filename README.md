# CoughVID

> Detecting COVID-19 by cough sounds.

- Used CoughVID dataset:
    - Link: https://zenodo.org/record/4498364#.YPK9tnUzaV4

## Requirements:
- PyTorch
- Librosa

## Steps:
### Setup
`$ bash setup.sh`
- It will download and prepare dataset
### Train
`$ python3 train.py`
### Test
`$ python3 test.py <audio file path>`