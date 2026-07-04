conda deactivate  
conda remove -n py312 --all       
conda create -n py312 python=3.12
conda activate py312      
conda install pytorch torchvision numpy wandb tqdm scikit-learn
pip install opencv-python --only-binary=opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
