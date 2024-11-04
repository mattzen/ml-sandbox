import torch
from diffusers import StableDiffusionPipeline

# Use StableDiffusionPipeline instead of FluxPipeline
model_name = "black-forest-labs/FLUX.1-dev"
pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)

# Move pipeline to GPU if available
if torch.cuda.is_available():
    pipeline.to("cuda")

# Define your text prompt
prompt = "A futuristic city skyline at sunset, high-tech and vibrant colors"

# Generate the image
image = pipeline(prompt, num_inference_steps=1).images[0]

# Display or save the image
image.show()  # To display
# image.save("output.png")  # To save