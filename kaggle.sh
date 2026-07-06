
############################################################
# ltcc
############################################################
# Lucky / ltcc  
# python main/main.py --config_file "main/config/method.yml" TASK.NAME=Lucky

accelerate launch --multi_gpu --num_processes 2 main/main.py --config_file "main/config/method.yml" TASK.NAME=Lucky