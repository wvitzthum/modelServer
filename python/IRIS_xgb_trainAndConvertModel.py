import numpy
import os
import onnxruntime as rt
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

# Load and prepare data
data = load_iris()
X = data.data[:, :2]  # Using only first two features
y = data.target
ind = numpy.arange(X.shape[0])
numpy.random.shuffle(ind)
X = X[ind, :].copy()
y = y[ind].copy()

# Create and train the pipeline
pipe = Pipeline([("scaler", StandardScaler()), ("xgb", XGBClassifier(n_estimators=3))])
pipe.fit(X, y)

# Register XGBoost converter
update_registered_converter(
    XGBClassifier,
    "XGBoostXGBClassifier",
    calculate_linear_classifier_output_shapes,
    convert_xgboost,
    options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
)

# Define input types for ONNX conversion
# Use shape [1, 1] instead of [1] to ensure proper dimensionality
input_types = [('sepal_length', FloatTensorType([1, 1])),
               ('sepal_width', FloatTensorType([1, 1]))]

# Convert to ONNX
model_onnx = convert_sklearn(
    pipe,
    "pipeline_xgboost",
    input_types,
    target_opset={"": 12, "ai.onnx.ml": 2},
    options={"zipmap": False}
)

# Create directory if it doesn't exist
os.makedirs("models/iris_xgboost/1", exist_ok=True)

# Save the model
model_path = os.path.join("models", "iris_xgboost", "1", "model.onnx")
with open(model_path, "wb") as f:
    f.write(model_onnx.SerializeToString())

print("Model saved to:", model_path)

# Compare predictions
print("\nComparing predictions:")
print("sklearn native:")
print("predict:", pipe.predict(X[:5]))
print("predict_proba:", pipe.predict_proba(X[:1]))

# Run ONNX inference
print("\nonnx conversion:")
sess = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Prepare input data for ONNX - process one sample at a time
input_data = X[:5].astype(numpy.float32)
all_predictions = []
all_probabilities = []

for i in range(len(input_data)):
    # Process one sample at a time
    inputs = {
        "sepal_length": input_data[i, 0].reshape(1, 1),  # Shape [1, 1]
        "sepal_width": input_data[i, 1].reshape(1, 1)    # Shape [1, 1]
    }
    
    # Run inference for this sample
    pred_onx = sess.run(None, inputs)
    all_predictions.append(pred_onx[0][0])
    if i == 0:  # Only store probability for first sample to match sklearn output
        all_probabilities.append(pred_onx[1])

# Convert to numpy arrays for comparison
all_predictions = numpy.array(all_predictions)
all_probabilities = numpy.array(all_probabilities)

print("predict:", all_predictions)
print("predict_proba:", all_probabilities)