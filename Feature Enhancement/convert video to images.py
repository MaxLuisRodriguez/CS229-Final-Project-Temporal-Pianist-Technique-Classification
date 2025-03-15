import cv2
import os
import numpy as np
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import pandas as pd

# Azure Storage account details
account_name = 'cvpiano1'
video_container = 'cs229advancedvideobatch5'
image_container = 'cs229advancedimagesequencesbatch5'

# Azure Access Key Connection string
connect_str = 'input access key string here...'

# Check connection to Azure Blob Storage
try:
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_service_client.get_account_information()
    print("Successfully connected to Azure Blob Storage.")
except Exception as e:
    print(f"Failed to connect to Azure Blob Storage: {e}")
    exit()

# container clients
video_container_client = blob_service_client.get_container_client(video_container)
image_container_client = blob_service_client.get_container_client(image_container)

# list video blobs
try:
    video_blob_list = [blob.name for blob in video_container_client.list_blobs() 
                      if blob.name.lower().endswith(('.mov', '.mp4'))]
    print(f"Found {len(video_blob_list)} video files: {video_blob_list}")
except Exception as e:
    print(f"Failed to list blobs in container {video_container}: {e}")
    exit()

# process each video
for blob_i in video_blob_list:
    try:
        # Generate a shared access signature for each blob file
        sas_i = generate_blob_sas(
            account_name=account_name,
            container_name=video_container,
            blob_name=blob_i,
            account_key=blob_service_client.credential.account_key,  # Use the key from the connection string
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
        
        sas_url = f'https://{account_name}.blob.core.windows.net/{video_container}/{blob_i}?{sas_i}'
        
        # Download the video file locally
        local_video_path = f"temp_{blob_i}"
        with open(local_video_path, "wb") as video_file:
            video_file.write(video_container_client.download_blob(blob_i).readall())
        print(f"Downloaded {blob_i} to {local_video_path} with size {os.path.getsize(local_video_path)} bytes")
        
        # open the video file
        cap = cv2.VideoCapture(local_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {blob_i}")
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_traversed = 0
        frame_interval = 4
        seq_num = 1 # for image label
        
        # create sequences of 20 chronologically ordered images with 10 frames between each
        while frames_traversed + (20 * frame_interval) <= total_frames:
            for i in range(20):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frames_traversed)  # Explicitly set frame position
                ret, frame = cap.read()
                if not ret:
                    print(f"Error: Could not read frame {frames_traversed} from {blob_i}")
                    break
                
                # Save the frame as an image
                image_name = f"{blob_i}_frame_{frames_traversed:04d}_seq_{seq_num}.jpg"
                local_image_path = f"temp_{image_name}"
                cv2.imwrite(local_image_path, frame)
                    
                # Upload the image to the image container
                with open(local_image_path, "rb") as image_file:
                    image_container_client.upload_blob(name=image_name, data=image_file, overwrite=True)
                print(f"Uploaded {image_name} to {image_container}")
                    
                frames_traversed += frame_interval
                os.remove(local_image_path)  # Clean up local image file
            seq_num += 1
        
        cap.release()
        os.remove(local_video_path)  # Clean up local video file

        print(f"Traversed {frames_traversed} frames from {blob_i} and uploaded to {image_container}.")
    except Exception as e:
        print(f"Error processing {blob_i}: {e}")

print("All videos processed.")