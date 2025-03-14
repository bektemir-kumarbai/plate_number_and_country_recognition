import grpc
from concurrent import futures
import sys
from pathlib import Path
import numpy as np
import os
import uuid
import tempfile
import cv2
from datetime import datetime
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)
from app.ai_service.app_gts import lp_det_reco
import plate_recognation_pb2
import plate_recognation_pb2_grpc
from PIL import Image
import io
from app.core.config import Settings

settings = Settings()
SECRET_TOKEN = settings.SECRET_TOKEN

# Create temp directory for storing images
TEMP_DIR = os.path.join(tempfile.gettempdir(), "license_plate_images")
os.makedirs(TEMP_DIR, exist_ok=True)

# Clean old temp files periodically (files older than 1 hour)
def clean_old_files():
    current_time = datetime.now().timestamp()
    one_hour = 3600  # seconds
    
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        if os.path.isfile(file_path):
            file_time = os.path.getmtime(file_path)
            if current_time - file_time > one_hour:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting temp file {file_path}: {e}")

class ImageProcessingServicer(plate_recognation_pb2_grpc.ImageProcessingServiceServicer):
    def ProcessImage(self, request, context):
        # Clean old files periodically
        clean_old_files()
        
        # Authenticate request
        if request.token != SECRET_TOKEN:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("Invalid authentication token")
            print("Invalid authentication token")
            return plate_recognation_pb2.ImageResponse()

        try:
            # Save image to temporary file
            image = Image.open(io.BytesIO(request.image_data))
            
            # Generate a unique filename with timestamp
            temp_filename = f"plate_{uuid.uuid4().hex}.jpg"
            temp_filepath = os.path.join(TEMP_DIR, temp_filename)
            
            # Save as JPG (more efficient than PNG)
            image.save(temp_filepath, "JPEG", quality=95)
            
            print(f"Image saved to {temp_filepath}")
            
            # Process the image using the file path
            prediction = lp_det_reco(temp_filepath)
            print(prediction)
            
            return plate_recognation_pb2.ImageResponse(
                license_plate_number=prediction['license_plate_number'],
                license_plate_number_score=prediction['license_plate_number_score'],
                license_plate_country=prediction['license_plate_country'],
                license_plate_country_score=prediction['license_plate_country_score']
            )

        except Exception as ex:
            print(f"Error processing image: {ex}")
            return plate_recognation_pb2.ImageResponse(
                license_plate_number=None,
                license_plate_number_score=None,
                license_plate_country=None,
                license_plate_country_score=None
            )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    plate_recognation_pb2_grpc.add_ImageProcessingServiceServicer_to_server(ImageProcessingServicer(), server)
    server.add_insecure_port("[::]:" + str(settings.APP_PORT))
    server.start()
    print(" gRPC server started on port " + str(settings.APP_PORT))
    print(f" Temporary images will be stored in {TEMP_DIR}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()