#!/bin/bash
# Model will be provided via volume mount or downloaded at runtime
# Example usage:
#   docker run -v /path/to/models:/app/models -p 8501:8501 debug-build
# Or place the model file at ./models/inswapper_128.onnx before building

MODEL_PATH="/app/models/inswapper_128.onnx"

if [ ! -f "$MODEL_PATH" ] || [ ! -s "$MODEL_PATH" ]; then
    echo "⚠️  Model not found at $MODEL_PATH"
    echo "Provide the model file in one of these ways:"
    echo "  1. Mount a volume: docker run -v /path/to/models:/app/models ..."
    echo "  2. Place ./models/inswapper_128.onnx before building the image"
    echo "  3. Download from: https://github.com/facefusion/facefusion-assets/releases"
    echo ""
    echo "Starting supervisor anyway - model download will be attempted at runtime by app..."
else
    echo "✓ Model found: $(ls -lh $MODEL_PATH)"
fi

# Start supervisor
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
