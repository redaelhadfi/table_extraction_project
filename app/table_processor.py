# app/table_processor.py
from PIL import Image, ImageDraw, ImageFont # Added ImageDraw, ImageFont for potential future use
import pandas as pd
import pytesseract
import torch
from app.models_loader import get_models_and_processor # Import from models_loader

# Attempt to get the globally loaded model and feature_extractor
try:
    model, feature_extractor = get_models_and_processor()
except RuntimeError as e:
    print(f"Critical error: {e}")
    # Fallback or raise error to prevent app from starting if models are crucial
    # For now, we'll let it proceed and it will fail on first use if not loaded.
    model, feature_extractor = None, None


def compute_boxes_from_image(pil_image):
    """
    Computes bounding boxes for table structures from a PIL Image.
    """
    if model is None or feature_extractor is None:
        raise ConnectionError("Model or feature extractor not loaded. Cannot process image.")

    encoding = feature_extractor(pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding)

    width, height = pil_image.size
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]
    
    boxes = results['boxes'].tolist()
    labels = results['labels'].tolist()
    scores = results['scores'].tolist()

    return boxes, labels, scores


def extract_table_from_image(pil_image):
    """
    Extracts table data into a Pandas DataFrame from a PIL image.
    Uses the globally loaded model and feature_extractor.
    """
    if model is None or feature_extractor is None:
        raise ConnectionError("Model or feature extractor not loaded. Cannot process image.")

    # Forcing the image to RGB, as the model expects 3 channels
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
        
    boxes, labels, _ = compute_boxes_from_image(pil_image) # We have scores if needed

    # Filter for table cells, rows, columns.
    # The original script's extract_table logic seems to rely on specific label indices
    # (e.g., label_row == 2 for rows, label_col == 1 for columns).
    # These indices might need to be verified against `model.config.id2label`.
    # For "microsoft/table-transformer-structure-recognition":
    #   'table': 0, 'table column': 1, 'table row': 2, 'table column header': 3,
    #   'table projected row header': 4, 'table spanning cell': 5
    # So, label 1 is 'table column' and label 2 is 'table row'.

    table_rows_boxes = [box for box, label in zip(boxes, labels) if label == 2] # table row
    table_cols_boxes = [box for box, label in zip(boxes, labels) if label == 1] # table column
    
    # Re-implementing the core logic of your `extract_table` function from the script:
    constructed_cell_locations = []
    for r_box, r_label in zip(boxes, labels):
        if r_label == 2: # 'table row'
            for c_box, c_label in zip(boxes, labels):
                if c_label == 1: # 'table column'
                    # Define cell: x from column, y from row.
                    cell_box = (c_box[0], r_box[1], c_box[2], r_box[3])
                    constructed_cell_locations.append(cell_box)
    
    if not constructed_cell_locations:
        print("No cell locations could be constructed from rows and columns.")
        return pd.DataFrame()

    constructed_cell_locations.sort(key=lambda x: (x[1], x[0])) # Sort by y_min, then x_min

    num_columns = 0
    if constructed_cell_locations:
        first_cell_y = constructed_cell_locations[0][1]
        # Count cells that are likely in the first row by checking y-coordinate similarity
        for cell_loc in constructed_cell_locations:
            if abs(cell_loc[1] - first_cell_y) < 5: # 5px tolerance for y alignment
                num_columns += 1
            else:
                # This assumes cells are sorted correctly and all first row cells come first
                break 
    
    if num_columns == 0 and constructed_cell_locations: # Handle case of a single row of cells
        num_columns = len(constructed_cell_locations)

    if num_columns == 0:
        print("Could not determine number of columns from constructed cells.")
        return pd.DataFrame()

    headers = []
    # Assuming the first `num_columns` sorted cells are headers if this logic is followed.
    # This might be fragile; direct header detection (e.g. label 'table column header') would be better.
    header_cells_to_ocr = constructed_cell_locations[:num_columns]
    
    for i, box in enumerate(header_cells_to_ocr):
        x1, y1, x2, y2 = map(int, box)
        if x1 >= x2 or y1 >= y2: continue 
        cell_image = pil_image.crop((x1, y1, x2, y2))
        try:
            # Using --psm 6 for assuming a single uniform block of text.
            cell_text = pytesseract.image_to_string(cell_image, config='--psm 6').strip()
        except pytesseract.TesseractError as e:
            print(f"Pytesseract error on header cell {i}: {e}")
            cell_text = ""
        headers.append(cell_text if cell_text else f"Header_{i+1}")

    df = pd.DataFrame(columns=headers if headers else [f"Column_{j+1}" for j in range(num_columns)])

    current_row_data = []
    data_cells_to_ocr = constructed_cell_locations[num_columns:]

    for i, box in enumerate(data_cells_to_ocr):
        x1, y1, x2, y2 = map(int, box)
        if x1 >= x2 or y1 >= y2: continue
        cell_image = pil_image.crop((x1, y1, x2, y2))
        try:
            cell_text = pytesseract.image_to_string(cell_image, config='--psm 6').strip()
        except pytesseract.TesseractError as e:
            print(f"Pytesseract error on data cell {i}: {e}")
            cell_text = ""
        
        current_row_data.append(cell_text)

        if len(current_row_data) == num_columns:
            if len(current_row_data) == len(df.columns):
                 # df.loc[len(df)] = current_row_data # Deprecated way
                df.loc[len(df.index)] = current_row_data
            else:
                print(f"Warning: Row data length {len(current_row_data)} mismatches df columns {len(df.columns)}. Row: {current_row_data}")
            current_row_data = []
    
    if current_row_data and len(current_row_data) == num_columns and len(df.columns) > 0 and len(current_row_data) == len(df.columns):
        df.loc[len(df.index)] = current_row_data
    elif current_row_data:
        print(f"Warning: Partial final row data {current_row_data} not added. Mismatched columns or no columns in df.")
        
    return df
