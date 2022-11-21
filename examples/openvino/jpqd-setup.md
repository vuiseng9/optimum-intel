### Setup to run JPQD on optimum-intel
Custom branches are needed until they are merged.
```bash
# Create Conda environment
conda create -n optimum-env python=3.8

# Install Optimum Intel
# goto working dir
git clone https://github.com/vuiseng9/optimum-intel
cd optimum-intel && git checkout nncf-jpqd
pip install -e .

# Install NNCF, OV
# goto working dir
git clone https://github.com/vuiseng9/nncf
cd nncf && git checkout p4-jpqd-dev
python setup.py develop

# For some reason the latest update of 1.11 ninja is causing error (as of Nov5 22)
pip install ninja==1.10.2.4

# Install dependent packages required in examples (we avoid requirements.txt in examples folder)
pip install torchvision evaluate wandb openvino

# OV IR serialization/generation will error out for prior version
pip install openvino-dev==2022.3.0.dev20221103

# openvino requires ~1.1.5 which fails to support df.to_markdown()
pip install --upgrade pandas

# Run movement sparsification on examples 
# there are bash (*.sh) scripts in examples/openvino/<task>/*.sh
```
