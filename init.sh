# Install sentencepiece
conda install -c conda-forge sentencepiece

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install g++
conda install gcc=12  -c conda-forge
conda install gxx=12  -c conda-forge

# Install openssl
conda install anaconda::openssl

# Install flash-attn
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp313-cp313-linux_x86_64.whl
