
############################################################
# ltcc
############################################################
# Lucky / ltcc  
accelerate launch --multi_gpu --num_processes 2 main/main.py --config_file "main/config/method.yml" TASK.NAME=Lucky TASK.NOTES=Debug