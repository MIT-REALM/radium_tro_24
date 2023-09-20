# Cuda
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
chmod +x ./cuda_12.1.1_530.30.02_linux.run
sudo ./cuda_12.1.1_530.30.02_linux.run --silent --override --toolkit --samples --toolkitpath=/usr/local/cuda-version --samplespath=/usr/local/cuda --no-opengl-libs
sudo ln -sf /usr/local/cuda-12.1 /usr/local/cuda

# Conda
conda init
source ~/.bashrc
conda create -n architect_env python=3.9 -y
conda activate architect_env
cd ~/src/architect_private && pip install -e . && pip install -r requirements.txt --ignore-requires-python
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
