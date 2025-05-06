python ffn_2_layer.py --profile_dir=/tmp/pytorch_profile --model_axis=2
gcloud storage cp --recursive /tmp/pytorch_profile gs://hanq_random/pytorch_profile_512
