docker run -t --rm -p 8100:8100 -v /Users/abhishek-pt5840/Desktop/College/mini-project/code/plant-disease/:/plant-disease tensorflow/serving --rest_api_port=8100 --model_config_file=plant-disease/models.config