# Setup TensorFlow development environment on Windows

1. Update NVIDIA driver to NVIDIA CUDA 10.2 or higher
2. Make sure the environment variables correctly contains the path to Visual Studio, CUDA, Python and pip.
Special note for laptops: If you have a laptop with an NVIDIA GPU then it should work. However, one unique problem on laptops is that you will likely have power saving control that switches your display driver back to the CPU's integrated display. A current Windows 10 setup on your laptop along with the latest driver should automatically switch your display to the NVIDIA driver when you start TensorFlow (same as starting up a game) but, if you have trouble that looks like TensorFlow is not finding your GPU then you may need to manually switch your display.
3. Download and install Anaconda Python (3.7 or newer version)
4. Start the Anaconda Powershell prompt and start installing/updating the following: (To exist the prompt: Ctrl+D)
* conda update conda
* conda update anaconda
* conda update python
* conda update --all
5. Create a Python "virtual environemtn for TensorFlow:
* conda create --name tf-gpu
6. Install Tensorflow-GPU from Anaconda Cloud repo
* conda install tensorflow-gpu
create a Jupyter Notebook kernel for the TensorFlow environment
* conda install ipykernel jupyter
python -m ipykernel install --user --name tf-gpu --display-name "TensorFlow-GPU-2.0.1"
Note: (start the notebook: jupyter notebook, under the "new" notebooks you should find the TensorFlow-GPU-2.0.1 empty notebook)

# 
