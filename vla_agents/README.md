# VLA as a tool
Setup and install openvla as described in https://github.com/openvla/openvla

```
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

# Clone and install the openvla repo
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

## Making a rephrased dataset:

```python make_rephrased_dataset.py```

## Running exploration on open VLA
Note sets up a local http server for the agent to communicate with the VLA. Also note that running openVLA requires 16gb of GPU memory and is highly reccomended to only run on an A100

```python exploration.py --task <task>```
Chose a task out of spatial, object, goal or 10

##
Evaluate after exploring

```python evaluate.py```
