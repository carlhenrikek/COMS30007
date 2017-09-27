1. Download 64-bit Linux Miniconda installation script for Python 3.6 from [here](https://conda.io/miniconda.html).
2. Run `sh <path-to-folder>/Miniconda3-latest-Linux-x86_64.sh`
3. Scroll through the license (press the space bar to move through quickly), type ‘yes’ to approve the terms, and then accept all the installation defaults.
4. Close the Terminal. Then, restart it.
5. Run `conda update conda` to update miniconda.
6. Run `conda create -n myenv python` to create a miniconda environment called "myenv".
5. Run `conda install -n myenv jupyter scipy numpy tensorflow matplotlib` to install the required packages in your environment.
7. Run `source activate myenv` to activate the miniconda environment.
7. Now you can run jupyter with `jupyter notebook`!

