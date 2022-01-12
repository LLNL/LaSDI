#!/bin/bash
python3 -m venv lasdi_venv
source lasdi_venv/bin/activate
pip install --upgrade pip
pip install --upgrade future
pip install mfem
pip install torch==1.7.1
pip install torchvision
pip install pysindy
pip install matplotlib
pip install tqdm
pip install path
pip install ipykernel
python -m ipykernel install --user --name lasdi_ipy --display-name "LaSDI-IPy"
pip install nbconvert
. mfem_setup.sh
