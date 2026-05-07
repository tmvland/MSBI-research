import os
import pydicom
import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, Dataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

def create_multiframe_dicom(input_dir, output_file):
    # Sort files, read, and convert to grayscale
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.dcm'))])
    # Load images, convert to grayscale, and convert to numpy array
    frames = [np.array(Image.open(os.path.join(input_dir, f)).convert('L')) for f in image_files]
    pixel_data_stack = np.array(frames, dtype=np.uint8)
    
    # Setup necessary DICOM metadata
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7' # Multi-frame
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Create Dataset
    ds = FileDataset(output_file, {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.Rows, ds.Columns = pixel_data_stack.shape[1], pixel_data_stack.shape[2]
    ds.NumberOfFrames = len(frames)
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.PixelData = pixel_data_stack.tobytes()
    
    ds.save_as(output_file)
    print(f"Saved: {output_file}")

# Example Usage: create_multiframe_dicom("input_images", "output.dcm")