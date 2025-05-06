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

gcloud compute tpus tpu-vm ssh --zone "us-central2-b" "ray-hanq-ray-cluster-worker-9914cfff-tpu" --project "tpu-pytorch" --worker=all --command="python -c 'import jax; print(jax.devices())'"
```

## Run docker
```
sudo docker run --net=host --privileged -it torchxla2
```


## xprof
8B, 2k cont length; 32xv4
https://xprof.corp.google.com/trace_viewer/hanq-12063596051371474787?hosts=t1v-n-df786e86-w-0&host_index=0&view_start=1380.712&view_end=2433.584