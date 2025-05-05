# modelServer
Small project to demonstrate how quickly KServe can be set up with a real use case scenario.
Currently, we serve two models that have been converted from sklearn pickle to ONNX and are embedded into a Triton Inference Server.
This project is currently only demoing a RawDeployment of KServe so no istio and Knative for scaling.

Future items to look into:
* Model Mesh
* Async Serving
* Knative/Istio scaling

Some high-level testing has shown 2.5- 3x increases in performance compared to BentoML/pickle while also drastically reducing the memory footprint.

## Instructions
### Prerequisites
* Make
* AWS CLI
* kind cluster running
* Python & Poetry


## Steps
### Install the kserve CRD on kind
```bash
# windows
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.3/cert-manager.yaml
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.15.0/kserve.yaml
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.15.0/kserve-cluster-resources.yaml
# linux
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.15/hack/quick_install.sh" | bash
```

## Deployment Mode
We have two options to deploy, the raw version, which is limited to just the inference servic,e so missing some of the following features:
* No automatic canary/rollout capabilities
* No built-in model versioning
* No auto-scaling based on inference metrics
* No pre/post-processing without custom implementation
* Missing model monitoring and explainability tools
  
### Serverless needs KNative to install it on the k8s cluster
```bash
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

### Alternatively, we can setup a Raw Deployment Mode
Add the following annotation to the deployment yaml(s).
```bash
metadata:
  annotations:
    serving.kserve.io/deploymentMode: "RawDeployment"
```

### Optional for K8s metrics
```bash
kubectl apply -f k8s-metrics.yaml
```

### Apply YAMLs to k8s cluster
The deploy folder contains the base (blob secrets etc) and overlay the inference components. Will add kustomize around this later on.
```bash
kubectl apply -f kserve-storage-config.yaml
kubectl apply -f kserve-inferenceserviceSecret.yaml
kubectl apply -f kserve-inferenceserviceSA.yaml

# please upload models first, otherwise this deploy will fail
kubectl apply -f kserve-inferenceservice.yaml
```

### Port Forward the blob storage so you can upload the model(s)
There is a makefile/ps1 script that can be used to auto-upload everything.
```bash
kubectl port-forward svc/minio-service 9000:9000 -n default
```

### List file tree on minio
```bash
aws --endpoint-url http://localhost:9000 --profile minio s3 ls s3://models/ --recursive
```


### If there's deployment issue,s the triton pod/service needs to be force deleted
```bash
kubectl delete pod sklearn-onnx-predictor-6bffbc89b7-5zx7c 
--grace-period=0 --force --namespace default
kubectl delete inferenceservice sklearn-onnx --grace-period=0 --force --namespace default
```


### Testing pbtxt configurations
The output of the following can be used to validate that the pbtxt files are in the correct shape. They can be a bit tricky to get right.
```bash
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v {PATH/TO/MODELS}\models:/models nvcr.io/nvidia/tritonserver:23.05-py

# in the docker container run 
tritonserver --model-repository=/models
```


## Testing the inference Server

### Port Forward the k8s service to 8000 localhost
```
kubectl port-forward svc/sklearn-onnx-predictor 8000:80
```

### Check readiness
```bash
curl http://localhost:8000/v2/health/ready
```

### Get model metadata
```bash
curl http://localhost:8000/v2/models/iris_xgboost/versions/1
```
returns
```json
{
	"name": "iris_xgboost",
	"versions": [
		"1"
	],
	"platform": "onnxruntime_onnx",
	"inputs": [
		{
			"name": "input",
			"datatype": "FP32",
			"shape": [
				2
			]
		}
	],
	"outputs": [
		{
			"name": "label",
			"datatype": "INT64",
			"shape": [
				2
			]
		},
		{
			"name": "probabilities",
			"datatype": "FP32",
			"shape": [
				2,
				3
			]
		}
	]
}
```
### Inference
Post requests can also be made against a specific version of the mode i.e.
http://localhost:8000/v2/models/wine_xgboost/versions/1/infer
The example below uses the default:
```bash
curl --request POST \
  --url http://localhost:8000/v2/models/wine_xgboost/infer \
  --header 'Content-Type: application/json' \
  --data '{
    "inputs": [
      {
        "name": "input",
        "shape": [5, 13],
        "datatype": "FP32",
        "data": [
          [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065],
          [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050],
          [12.37, 1.17, 1.92, 19.6, 78, 2.11, 2.0, 0.27, 1.04, 4.68, 1.12, 3.48, 510],
          [12.33, 1.1, 2.28, 16.0, 101, 2.05, 1.09, 0.63, 0.41, 3.27, 1.25, 1.67, 680],
          [13.36, 2.56, 2.35, 20.0, 89, 1.4, 0.5, 0.37, 0.64, 3.05, 0.64, 1.93, 750]
        ]
      }
    ]
  }'
```
This should give us back something akin to
```json
{
	"model_name": "wine_xgboost",
	"model_version": "1",
	"outputs": [
		{
			"name": "label",
			"datatype": "INT64",
			"shape": [
				5
			],
			"data": [
				0,
				0,
				1,
				1,
				1
			]
		},
		{
			"name": "probabilities",
			"datatype": "FP32",
			"shape": [
				5,
				3
			],
			"data": [
				0.9957030415534973,
				0.001987984636798501,
				0.002308981027454138,
				0.9954540729522705,
				0.002237503184005618,
				0.00230840384028852,
				0.008922455832362175,
				0.9808900356292725,
				0.010187500156462193,
				0.0018183544743806124,
				0.9852442741394043,
				0.012937391176819802,
				0.013885765336453915,
				0.7449164390563965,
				0.24119773507118226
			]
		}
	]
}
  
```

## Sample Metrics
```bash
kubectl top pod -n default
NAME                                              CPU(cores)   MEMORY(bytes)   
minio-599d8b578c-nzrwd                            1m           241Mi
sklearn-onnx-predictor-6dbbf5957-qtxdf            1m           47Mi
wine-xgboost-predictor-8f69c77db-spkgn            1m           34Mi
wine-xgboost-predictor-default-66b75f46cd-6v5ss   1m           32Mi

```
