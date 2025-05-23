import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
from pathlib import Path

load_dotenv()

def encode_image(image_path):
    """
    Encode image to base64 and determine the correct MIME type
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Get file extension and determine MIME type
    file_extension = Path(image_path).suffix.lower()
    mime_type = "image/jpeg" if file_extension in ['.jpg', '.jpeg'] else "image/png"
    
    return encoded_image, mime_type

def analyze_images(image_paths, prompt):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    with open(prompt, 'r') as f:
        prompt_text = f.read()
    
    encoded_images = [encode_image(path) for path in image_paths]
    encode_image_1 = encode_image(image_paths[0])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                *[{"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img}"}} 
                  for img, mime_type in encoded_images]
            ]
        }
    ]

    print(messages)
    
    # Hacer la llamada a la API
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        max_tokens=1000
    )
    
    return response

if __name__ == "__main__":
    image_paths = [
        "images/image1.png",  
        "images/image2.png"   
    ]
    prompt_path = "prompts/compare_images.txt"
    
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Error: Image {path} not found")
            exit(1)
    
    result = analyze_images(image_paths, prompt_path)
    print(result)