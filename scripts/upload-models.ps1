# PowerShell script to upload an ONNX model to MinIO (S3 compatible)

# === Configuration ===
$Bucket = "models"
$BucketFolder = ""
$ModelFolder = "models"
$EndpointUrl = "http://localhost:9000"  # Change to http://minio.minio:9000 if using in-cluster
$AccessKey = "minioadmin"
$SecretKey = "minioadmin"

# === Set environment for AWS CLI ===
$env:AWS_ACCESS_KEY_ID = $AccessKey
$env:AWS_SECRET_ACCESS_KEY = $SecretKey

# === Create bucket if it doesn't exist ===
Write-Host "Creating bucket '$Bucket' (if it doesn't exist)..."
aws --endpoint-url $EndpointUrl s3 mb "s3://$Bucket" 2>$null

# === Upload model folder ===
Write-Host "Uploading '$ModelFolder' to s3://$Bucket/$BucketFolder/..."
aws --endpoint-url $EndpointUrl s3 cp $ModelFolder "s3://$Bucket/$BucketFolder" --recursive

Write-Host "✅ Upload complete."
