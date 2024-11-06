sudo docker build --network=host -t llama3 .
sudo docker tag llama3 gcr.io/tpu-pytorch/llama3:latest
sudo docker push gcr.io/tpu-pytorch/llama3:latest
