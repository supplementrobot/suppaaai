## Supplement

- System requirement

  ```
  python >= 3.8
  CUDA >= 11.0
  ```

- Other main requirements

  PyTorch

  ```
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

  MinkowskiEngine

  ```
  conda install openblas-devel -c anaconda
  pip install pip==22.3.1
  pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
  ```

  pytorch-metric-learning

  ```
  pip install pytorch-metric-learning==1.1
  ```

- Datasets and weights downloading

  We currently prepare the Boreas dataset, which can be downloaded from:

  https://drive.google.com/file/d/1P_YH0d7CPtZaEAoii9FBpNda2rXIZVvh/view?usp=sharing

  The pre-trained teacher weights can be downloaded from:

  https://drive.google.com/file/d/1D6D4fPFrSG6r9m_yT-y-cjjw2KfWYVZK/view?usp=sharing

  After downloading the datasets and the weights, you need to unzip them. Then you need to replace the corresponding arguments in `tools/options.py`, including:

   `--teacher_weights_path`, `--dataset_folder`, and `--image_path`

- Train/test

  Then you can run the script to conduct distillation.

  ```
  python train.py
  ```

  

