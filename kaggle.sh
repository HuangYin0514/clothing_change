
############################################################
# ltcc
############################################################
# 调试
accelerate launch --multi_gpu --num_processes 2 main/main.py \
--config_file "main/config/method.yml" \
TASK.NOTES=v19 TASK.NAME=B OPTIMIZER.TOTAL_TRAIN_EPOCH=1 TEST.EVAL_EPOCH=1 TEST.SAVE_START_EPOCH=-1

# 基线
accelerate launch --multi_gpu --num_processes 2 main/main.py \
--config_file "main/config/method.yml" \
TASK.NOTES=v21 TASK.NAME=B 

# 可视化
python main/vis_main.py --config_file "main/config/method.yml"