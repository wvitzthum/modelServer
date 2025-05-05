import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
import onnxruntime as rt
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

# Load the Wine dataset
data = load_wine()
X = data.data  # 13 features
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create pipeline
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(n_estimators=50, max_depth=6, random_state=42)),
    ]
)

# Train
pipe.fit(X_train, y_train)

update_registered_converter(
    XGBClassifier,
    "XGBoostXGBClassifier",
    calculate_linear_classifier_output_shapes,
    convert_xgboost,
    options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
)

# Convert to ONNX (no batch dimension)
onnx_model = convert_sklearn(
    pipe,
    "pipeline_xgboost",
    initial_types=[("input", FloatTensorType([13]))],  # No batch dim
    target_opset={"": 12, "ai.onnx.ml": 2},
    options={"zipmap": False},
)

# Save
with open("wine_xgboost_single.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())


print("comparing predictions:")
print("sklearn native:")
# Inference test
print("Sklearn prediction:", pipe.predict(X_test[:5]))
print("Sklearn proba:", pipe.predict_proba(X_test[:1]))

print("onnx conversion:")
sess = rt.InferenceSession("wine_xgboost_single.onnx", providers=["CPUExecutionProvider"])
pred_onx = sess.run(None, {"input": X_test[:5].astype(np.float32)})
print("predict", pred_onx[0])
print("predict_proba", pred_onx[1][:1])