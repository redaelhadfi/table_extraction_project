# app/models_loader.py
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
import torch

print("Loading models and processor...")

# Common feature extractor/processor
try:
    feature_extractor = DetrImageProcessor() # Or DetrFeatureExtractor, ensure consistency
except Exception as e:
    print(f"Error loading DetrImageProcessor: {e}")
    print("Attempting to load DetrFeatureExtractor as a fallback...")
    from transformers import DetrFeatureExtractor
    feature_extractor = DetrFeatureExtractor()


# Model for table structure recognition (used in extract_table)
try:
    structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    structure_model.eval() # Set to evaluation mode
except Exception as e:
    print(f"Error loading structure_model: {e}")
    structure_model = None

# Model for table detection (if you want to add an endpoint for just detection)
# detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
# detection_model.eval()

print("Models and processor loaded.")

def get_models_and_processor():
    if structure_model is None or feature_extractor is None:
        raise RuntimeError("Models or feature_extractor could not be loaded. Check logs.")
    return structure_model, feature_extractor
