# modelServer

## Instructions

Use the supplied makeFile to upload the model files


### Prerequisites
* Make
* AWS cli
* kind cluster running
* 

## Install the kserve CRD on kind
```
# windows
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.3/cert-manager.yaml
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.15.0/kserve.yaml
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.15.0/kserve-cluster-resources.yaml
# linux
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.15/hack/quick_install.sh" | bash
```

## Deployment Mode

### Serverless needs Knative, to install it on the k8s cluster
```
# Install Knative Serving
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.12.0/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.12.0/serving-core.yaml

# Install Kourier (default ingress)
kubectl apply -f https://github.com/knative/net-kourier/releases/download/knative-v1.12.0/kourier.yaml

# Configure Knative to use Kourier
kubectl patch configmap/config-network \
  --namespace knative-serving \
  --type merge \
  --patch '{"data":{"ingress.class":"kourier.ingress.networking.knative.dev"}}'
```

### Alternatively we can setup a Raw Deployment Mode
Add the following annotation to the deployment yaml(s).
```
metadata:
  annotations:
    serving.kserve.io/deploymentMode: "RawDeployment"
```

### Optional for K8s metrics
```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

## Apply yamls to k8s cluster
```
kubectl apply -f kserve-inferenceservice.yaml
kubectl apply -f kserve-storage-config.yaml
```

## Port Forward the blob storage so you can upload the model(s)
```
kubectl port-forward svc/minio-service 9000:9000 -n default
```

## List file tree on minio
```
aws --endpoint-url http://localhost:9000 --profile minio s3 ls s3://models/ --recursive
```


## If theres deployment issues the triton pod needs to be force deleted
```
kubectl delete pod sklearn-onnx-predictor-6bffbc89b7-5zx7c 
--grace-period=0 --force --namespace default
```


## Testing pbtxt configurations
The output of the following can be used to validate the pbtxt files are in correct shape. They can be a bit tricky to get right.
```
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v {PATH/TO/MODELS}\models:/models nvcr.io/nvidia/tritonserver:23.05-py

# in the docker container run 
tritonserver --model-repository=/models
```
