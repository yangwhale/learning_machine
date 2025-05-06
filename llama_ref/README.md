## Start a cluster

`ray up cluster.yaml`

Useful commands:
  To terminate the cluster:
    ray down cluster.yaml

  To retrieve the IP address of the cluster head:
    ray get-head-ip cluster.yaml

  To port-forward the cluster's Ray Dashboard to the local machine:
    ray dashboard cluster.yaml

  To submit a job to the cluster, port-forward the Ray Dashboard in another terminal and run:
    ray job submit --address http://localhost:<dashboard-port> --working-dir . -- python my_script.py

  To connect to a terminal on the cluster head for debugging:
    ray attach cluster.yaml

  To monitor autoscaling:
    ray exec cluster.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'

## Build docker
```
sudo docker build --network=host -t llama3 .
sudo docker tag llama3 gcr.io/tpu-pytorch/llama3:latest
sudo docker push gcr.io/tpu-pytorch/llama3:latest
```

```
gcloud compute tpus tpu-vm ssh --zone "us-central2-b" "ray-hanq-ray-cluster-worker-b3686b6a-tpu" --project "tpu-pytorch" --worker=all --command="sudo docker run --net=host --pull=always --privileged gcr.io/tpu-pytorch/llama3:latest"
```

## Run docker
```
sudo docker run --net=host --privileged -it torchxla2
```