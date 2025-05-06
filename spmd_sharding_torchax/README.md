# Run locally:

```
python ffn_2_layer.py
```

# Run on xpk

NOTE: edit the file for region / project / cluster id
then

```
./run_xpk.sh
```

It should print out a link to see the logs : https://pantheon.corp.google.com/kubernetes/service/europe-west4/mlperf-v5p-128/default/hanq-1-20250124-202020/logs?e=13802955&inv=1&invt=Abnuyw&mods=allow_workbench_image_override&project=cloud-tpu-multipod-dev

The profile should be in this bucket: https://pantheon.corp.google.com/storage/browser/hanq_random/pytorch_tpu_profile/plugins/profile;tab=objects?e=13803378&inv=1&invt=AbnuwA&mods=allow_workbench_image_override&project=cloud-tpu-multipod-dev&prefix=&forceOnObjectsSortingFiltering=false

Pick the datetime that is most recent