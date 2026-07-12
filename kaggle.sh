############################################################
# ltcc
############################################################
# OPTIMIZER.TOTAL_TRAIN_EPOCH=1 TEST.EVAL_EPOCH=1 TEST.SAVE_START_EPOCH=-1 # 调试

# 基线（R50）
accelerate launch --multi_gpu --num_processes 2 main/main.py \
--config_file "main/config/method.yml" \
TASK.NOTES=v25 TASK.NAME=B MODEL.BACKBONE_TYPE=resnet50

# 基线（R50_IBN）
accelerate launch --multi_gpu --num_processes 2 main/main.py \
--config_file "main/config/method.yml" \
TASK.NOTES=v26 TASK.NAME=B_IBN MODEL.BACKBONE_TYPE=resnet50_ibn_a

############################################################
# 可视化
############################################################
# 可视化
python main/vis_main.py --config_file "main/config/method.yml"

# 生成网格可视化预览
%cd /kaggle/working/clothing_change
%matplotlib inline
!python main/util/display_images_from_dir.py --img_dir=/kaggle/working/clothing_change/results/actmap \
--max_imgs=30 --rows=5 --cols=6
from IPython.display import Image, display
display(Image("/kaggle/working/clothing_change/results/img_grid/image_grid_output.png"))