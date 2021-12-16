# create env from file
conda env create --name miniworld -f env.yml
conda activate miniworld

pip install -e .