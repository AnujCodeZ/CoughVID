wget -O data.zip https://zenodo.org/record/4498364/files/public_dataset.zip?download=1
unzip data.zip
rm -rf data.zip
python3 prepare_dataset.py