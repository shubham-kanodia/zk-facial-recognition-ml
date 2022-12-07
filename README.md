# Instructions

Download the Yale Face Database from https://www.kaggle.com/datasets/olgabelitskaya/yale-face-database

- Extract the downloaded data and put the images in data/yale-dataset folder
- Create data/processed-images folder
- `python3 pre_process_dataset.py` (cd training)
- Create data/processed-dataset folder
- `python3 data_prep.py` (cd training)
- Create model folder
- `python3 train.py` (cd training)
- Create data/snark-data folder
- `python3 populate_model_weights.py` (cd zk)

**Note:**
Run the below line if you face the no module named "xyz" issue. It should solve it in most cases.
export PYTHONPATH=$PYTHONPATH:`pwd`
