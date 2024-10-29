import re
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import pandas as pd  # Import pandas for DataFrame

# Load pre-trained processor and model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Prepare decoder inputs
task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
question = "What is the Vendor Name?"
prompt = task_prompt.replace("{user_input}", question)
decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

# Process the image
image = Image.open("Sample_Image.jpg")
pixel_values = processor(image, return_tensors="pt").pixel_values

# Generate outputs
outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

# Decode and clean up the output
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove first task start token

print(processor.token2json(sequence))

# Prepare data for Excel
data = {
    "Extracted Text": [sequence]  # Use sequence as the extracted text
}

# Create a DataFrame from the data dictionary
df = pd.DataFrame(data)

# Define the path to save the Excel file
excel_path = r"c:\\Users\\arcks\\Downloads\\OCRextractedtext.xlsx"  # Use raw string for Windows path

# Write the DataFrame to an Excel file
df.to_excel(excel_path, index=False)

print(f"Extracted text has been written to {excel_path}")
