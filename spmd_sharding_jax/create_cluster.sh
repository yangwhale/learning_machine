PROJECT=cloud-tpu-multipod-dev 
ZONE=europe-west4-b
TPU_TYPE=v5p-2048
NUM_SLICES=1
reservation=cloudtpu-20240716121201-595617744
CLUSTER_NAME=hanq-1024


gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}

# Create network settings based on: https://github.com/google/maxtext/blob/main/MaxText/configs/README.md
# This only needs to be created once per project!
NETWORK_NAME=hanq-testing-mtu9k
FIREWALL_NAME=hanq-testing-mtu9kfw

#CLUSTER_NAME=perf-v5p-4096
REGION=europe-west4
NETWORK_NAME=${CLUSTER_NAME}-$ZONE
FIREWALL_NAME=${CLUSTER_NAME}-$ZONE-fw
SUBNET_NAME=${NETWORK_NAME}-sub

# One time per cluster commands
# gcloud compute networks create "${NETWORK_NAME}" --mtu=8896 --bgp-routing-mode=regional --subnet-mode=custom --project="${PROJECT}"
# gcloud compute networks subnets create "${SUBNET_NAME}" --network="${NETWORK_NAME}" --range=10.10.0.0/18 --region="${REGION}" --project="${PROJECT}"
# gcloud compute firewall-rules create "${FIREWALL_NAME}" --network "${NETWORK_NAME}" --allow tcp,icmp,udp --project="${PROJECT}"
#--default_pool_cpu_machine_type=n2-standard-16
#--gke-version=1.30.3-gke.1969000
CLUSTER_ARGUMENTS="--network=mtu9k --subnetwork=mtu9k"

xpk cluster create --cluster $CLUSTER_NAME --num-slices=$NUM_SLICES --tpu-type=$TPU_TYPE --zone=$ZONE --project=$PROJECT --reservation=$reservation  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}"
