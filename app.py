import os

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from prometheus_fastapi_instrumentator import Instrumentator


from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('opentelemetry.sdk.trace').setLevel(logging.DEBUG)



# Define the CNN model architecture (same as the one used during training)
IN_CHANNELS = 1
FILTERS = 32
DROPOUT = 0.4

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(IN_CHANNELS, FILTERS, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(FILTERS)
        self.conv2 = nn.Conv2d(FILTERS, FILTERS, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(FILTERS)
        self.conv3 = nn.Conv2d(FILTERS, FILTERS * 2, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(FILTERS * 2)
        self.conv4 = nn.Conv2d(FILTERS * 2, FILTERS * 2, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(FILTERS * 2)
        self.conv5 = nn.Conv2d(FILTERS * 2, FILTERS * 2, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(FILTERS * 2)
        self.conv6 = nn.Conv2d(FILTERS * 2, FILTERS * 2, kernel_size=5, stride=2)
        self.bn6 = nn.BatchNorm2d(FILTERS * 2)
        self.dropout_conv = nn.Dropout(DROPOUT)
        self.fc1 = nn.Linear(FILTERS * 2, 128)
        self.dropout_fc = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.dropout_conv(x)
        x = x.view(-1, FILTERS * 2)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

# Load the model and state_dict
MODEL_PATH = "best_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # Set the model to evaluation mode

# Define the image transformation (same as in training)
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Ensure the image is resized to 28x28
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

def process_image(image_bytes):
    """
    Process the image into the correct format for the model.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert image to grayscale
    image = transform(image)  # Apply transformations: Resize, ToTensor, and Normalize
    image = image.unsqueeze(0)  # Add batch dimension as expected by the model
    return image

# API Key setup
API_KEY = os.getenv("API_KEY", "default_api_key")
API_KEY_NAME = "api_key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate API key",
        )
    

# Initialize FastAPI app
app = FastAPI()
# Initialize the Prometheus Instrumentator
instrumentator = Instrumentator()
# Register the metrics endpoint with Prometheus Instrumentator
instrumentator.instrument(app).expose(app, include_in_schema=False)


# Set up OpenTelemetry tracing
resource = Resource.create(attributes={"service.name": "fastapi-ml-service"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer_provider = trace.get_tracer_provider()

# Configure the OTLP exporter for sending trace data to the OpenTelemetry Collector
otlp_exporter = OTLPSpanExporter(
    endpoint="http://otel-collector:4321", 
    insecure=True
)
span_processor = SimpleSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)

# Instrument the FastAPI app with OpenTelemetry for automatic tracing
FastAPIInstrumentor.instrument_app(app)

# Instrument other libraries like logging and requests
LoggingInstrumentor().instrument()
RequestsInstrumentor().instrument()



@app.post("/predict/")
async def predict(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    # Log the API call
    logger.info("Prediction API called")
    
    try:
        # Read the image file uploaded by the client
        image_bytes = await file.read()
        image_tensor = process_image(image_bytes)  # Process the image for inference
        
        # Perform inference using the trained model
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
        logger.info(f"Predicted digit: {predicted.item()}")  # Log the prediction
        return {"predicted_digit": predicted.item()}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")