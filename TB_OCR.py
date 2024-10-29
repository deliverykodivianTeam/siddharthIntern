import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests

model_id = "yifeihu/TB-OCR-preview-0.1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map="cuda", 
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='flash_attention_2'
)

processor = AutoProcessor.from_pretrained(model_id, 
  trust_remote_code=True, 
  num_crops=16
)

def phi_ocr(image):
    question = "What is Vendor name?" # this is required
    image = Image.open(image)
    prompt_message = [{
        'role': 'user',
        'content': f'<|image_1|>\n{question}',
    }]

    prompt = processor.tokenizer.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda") 

    generation_args = { 
        "max_new_tokens": 1024, 
        "temperature": 0.1, 
        "do_sample": False
    }

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
    )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

    response = response.split("<image_end>")[0] # remove the image_end token 

    return response


response = phi_ocr("Sample_Image.jpg")

print(response)