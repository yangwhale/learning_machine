# Run locally:

```
python ffn_jax.py --model_axis 4 --num_layers 48 --profile_path "gs://lsiyuan-multipod-2/lightricks-ffn-profile/local-v5e-8-tmp"
```

# Run on xpk

NOTE: edit the file for region / project / cluster id
then

```
./run_xpk.sh
```

It should print out a link to see the logs : https://pantheon.corp.google.com/kubernetes/service/europe-west4/mlperf-v5p-128/default/hanq-1-20250124-202020/logs?e=13802955&inv=1&invt=Abnuyw&mods=allow_workbench_image_override&project=cloud-tpu-multipod-dev


Pick the datetime that is most recent