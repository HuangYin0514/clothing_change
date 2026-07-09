
############################################################
# ltcc
############################################################
# Lucky / ltcc  
accelerate launch --multi_gpu --num_processes 2 main/main.py --config_file "main/config/method.yml" TASK.NOTES=v18 TASK.NAME=Lucky 

# python main/vis_main.py --config_file "main/config/method.yml"  MODEL.TEST.RESUME_TEST_MODEL=115 
