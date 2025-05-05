import numpy as np
from sklearn.datasets import load_wine  # or load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
import onnxruntime as rt
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

# Load dataset (Wine or Iris)
data = load_wine()  # or load_iris()
X = data.data  # For Wine: 13 features; For Iris you may want X = data.data[:, :2]
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

# Option 1: Convert with fixed batch dimension for Triton
# This is the key change - set batch size to a fixed number (like 5)
num_features = X.shape[1]  # 13 for Wine, 2 for Iris subset
fixed_batch_size = 5       # Fixed batch size of 5

onnx_model = convert_sklearn(
    pipe,
    "pipeline_xgboost",
    [("input", FloatTensorType([fixed_batch_size, num_features]))],  # Fixed batch size
    target_opset={"": 12, "ai.onnx.ml": 2},
    options={"zipmap": False},
)

# Save
with open("models\wine_xgboost\\1\model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("comparing predictions:")
print("sklearn native:")
# Inference test
print("Sklearn prediction:", pipe.predict(X_test[:fixed_batch_size]))
print("Sklearn proba:", pipe.predict_proba(X_test[:1]))

print("onnx conversion:")
sess = rt.InferenceSession("models\wine_xgboost\\1\model.onnx", providers=["CPUExecutionProvider"])

# Must use exactly the same batch size as defined in the model
test_batch = X_test[:fixed_batch_size].astype(np.float32)
# Ensure we have exactly the right batch size
if test_batch.shape[0] < fixed_batch_size:
    # Pad with duplicates if needed
    padding = np.repeat(test_batch[-1:], fixed_batch_size - test_batch.shape[0], axis=0)
    test_batch = np.vstack([test_batch, padding])

pred_onx = sess.run(None, {"input": test_batch})
print("predict", pred_onx[0])
print("predict_proba", pred_onx[1][:1])

# To use in production with different batch sizes, you would need to:
# 1. Either pad smaller batches to the fixed size
# 2. Or create multiple models with different batch sizes
# 3. Or use option 2 below for individual sample processing

# Option 2: Process samples individually
# If you need to process variable numbers of samples with a model that
# expects single samples, you could use:

def process_samples_individually(samples, session):
    results = []
    probas = []
    
    for i in range(samples.shape[0]):
        single_sample = samples[i:i+1].astype(np.float32)  # Keep 2D but single row
        pred = session.run(None, {"input": single_sample})
        results.append(pred[0])
        probas.append(pred[1])
    
    return np.vstack(results), np.vstack(probas)

# Example usage:
# predictions, probabilities = process_samples_individually(X_test[:10], sess)