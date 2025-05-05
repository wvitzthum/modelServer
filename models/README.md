### Instruction

## Folder Structure
In order to make our models usable with extensive metadata we need them in ONNX format with a folder structure conforming with Triton structures. Please refer to the following link for an overview https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html.

Example blow with the xgboost sample model of version 1:
```
iris_xgboost/
├── 1/
│   └── model.onnx
└── config.pbtxt
```