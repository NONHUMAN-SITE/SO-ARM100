import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
from pathlib import Path
import numpy as np
from PIL import Image
import io
from enum import Enum
from pydantic import BaseModel, Field
from soarm100.logger import logger

load_dotenv()

class ValidResponse(Enum):
    OK = "Task finished successfully"
    INCOMPLETE = "Task is not finished. In process"
    ERROR = "Task failed"

class ValidResponseSchema(BaseModel):
    response: ValidResponse = Field(description="The response to the task")

class SupervisorAgent:
    def __init__(self, system_prompt=None):
        """
        Initialize the SupervisorAgent with OpenAI client and optional system prompt
        
        Args:
            system_prompt (str, optional): System prompt to use for the agent
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Default system prompt if none provided
        self.system_prompt = system_prompt or (
            "You are an assistant that answers queries about the status of a robot's actions.",
            "You will be given a question, the current task assigned to the robot, and an image representing the current moment.",
            "Based on this information, you must respond only whether the robot has completed the task, is currently in progress, or has failed."
        )

    def _encode_image_file(self, image_path):
        """
        Encode image from file path to base64 and determine MIME type
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (encoded_image, mime_type)
        """
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        file_extension = Path(image_path).suffix.lower()
        mime_type = "image/jpeg" if file_extension in ['.jpg', '.jpeg'] else "image/png"
        
        return encoded_image, mime_type

    def _encode_numpy_array(self, image_array):
        """
        Encode numpy array image to base64
        
        Args:
            image_array (numpy.ndarray): Image as numpy array
            
        Returns:
            tuple: (encoded_image, mime_type)
        """
        # Convert to uint8 if not already
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Convert to RGB if grayscale
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        
        # Convert to PIL Image
        image = Image.fromarray(image_array)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Encode to base64
        encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')
        return encoded_image, "image/png"

    def _process_image(self, image):
        """
        Process image whether it's a file path or numpy array
        
        Args:
            image: Either a string path to image or numpy array
            
        Returns:
            tuple: (encoded_image, mime_type)
        """
        if isinstance(image, str):
            return self._encode_image_file(image)
        elif isinstance(image, np.ndarray):
            return self._encode_numpy_array(image)
        else:
            raise ValueError("Image must be either a file path (str) or numpy array")

    def analyze(self, images, prompt):
        """
        Analyze images with given prompt
        
        Args:
            images (list): List of images (can be file paths or numpy arrays)
            prompt (str): Text prompt for analysis
            
        Returns:
            dict: OpenAI API response
        """

        logger.log(f"Prompt: {prompt}", level="success")
        logger.log(f"Images: {images}", level="success")
        
        if os.path.exists(prompt):
            with open(prompt, 'r') as f:
                prompt_text = f.read()
        else:
            prompt_text = prompt
        
        encoded_images = [self._process_image(img) for img in images]
        
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    *[{"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img}"}} 
                      for img, mime_type in encoded_images]
                ]
            }
        ]
        
        #response = self.client.beta.chat.completions.parse(
        #    model="gpt-4.1-mini",
        #    messages=messages,
        #    response_format=ValidResponseSchema
        #)
        #return response.choices[0].message.parsed.response.value

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=1000
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    # Example usage
    agent = SupervisorAgent()
    
    # Test with file paths
    image_paths = [
        "test/multimodal/images/image1.png",
        "test/multimodal/images/image2.png"
    ]
    
    # Verify images exist
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Error: Image {path} not found")
            exit(1)
    
    # Test with prompt file
    prompt_path = "test/multimodal/prompts/compare_images.txt"
    result = agent.analyze(image_paths, prompt_path)
    print("Result from file paths:", result)
    
    # Test with numpy arrays (example)
    try:
        # Create sample numpy arrays (100x100 random images)
        numpy_images = [
            np.random.rand(100, 100, 3),  # RGB image
            np.random.rand(100, 100)      # Grayscale image
        ]
        result = agent.analyze(numpy_images, "Compara estas dos imágenes aleatorias")
        print("\nResult from numpy arrays:", result)
    except Exception as e:
        print(f"Error testing numpy arrays: {e}")