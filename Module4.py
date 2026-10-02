import torch
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image




class Module4:

    def __init__(self):

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = self.pipe.to(self.device)


    def generate(self, image, Mfinal, prompt, same_valence=False):
        """
        image: PIL Image, the original source image
        Mfinal: numpy array (H, W), values 0-1, from Module 2's CAM_and_SAM output
        prompt: the templated motif string, e.g. "A joyful child enjoying a meal"
        same_valence: True if enhancing the existing valence, False if flipping it
        """
        img_512 = image.resize((512, 512))

        # --- Recall from the Module 2 paper text: which region actually gets edited
        # depends on same-valence vs. cross-valence editing ---
        if same_valence:
            Medit = 1 - Mfinal   # preserve the affective region, edit the SURROUNDING context
        else:
            Medit = Mfinal       # directly edit the affective region itself

        # --- Convert the numpy mask into a proper grayscale PIL mask ---
        mask_array = (Medit * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_array).resize((512, 512))

        negative_prompt = "low quality, blurry, distorted"

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img_512,
            mask_image=mask_image,
            num_inference_steps=50,
            guidance_scale=7.5,
        ).images[0]

        return result