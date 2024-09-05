# Image-Generator

Image generated from this model and code is available on https://drive.google.com/drive/folders/1Dw_B4iNsREdNk2tPA25-b2hAWRURMUOi?usp=sharing



## Step to run this model ##


Step-1:
### install requirements ###

!pip install --upgrade diffusers[torch]
!pip install transformers



Step-2:
### create image generation pipeline ###
Go to Huggingface-Langchain.com --> Go to model tab --> Search "rupeshs/LCM-runwayml-stable-diffusion-v1-5" --> Copy the model paste in the program

from diffusers import DiffusionPipeline
import torch


pipeline = DiffusionPipeline.from_pretrained("rupeshs/LCM-runwayml-stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

Step-3:
### generate images ###

        import random
        import os
        
        import matplotlib.pyplot as plt
        
        
        os.makedirs('/content/faces/happy', exist_ok=True)
        os.makedirs('/content/faces/sad', exist_ok=True)
        os.makedirs('/content/faces/angry', exist_ok=True)
        os.makedirs('/content/faces/surprised', exist_ok=True)
        
        
        ethnicities = ['a latino', 'a white', 'a black', 'a middle eastern', 'an indian', 'an asian']
        
        genders = ['male', 'female']
        
        emotion_prompts = {'happy': 'smiling',
                           'sad': 'frowning, sad face expression, crying',
                           'surprised': 'surprised, opened mouth, raised eyebrows',
                           'angry': 'angry'}
        
        
        for j in range(250):
        
          for emotion in emotion_prompts.keys():
        
            emotion_prompt = emotion_prompts[emotion]
        
            ethnicity = random.choice(ethnicities)
            gender = random.choice(genders)
        
            # print(emotion, ethnicity, gender)
        
            prompt = 'Medium-shot portrait of {} {}, {}, front view, looking at the camera, color photography, '.format(ethnicity, gender, emotion_prompt) + \
                    'photorealistic, hyperrealistic, realistic, incredibly detailed, crisp focus, digital art, depth of field, 50mm, 8k'
            negative_prompt = '3d, cartoon, anime, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ' + \
                              '((grayscale)) Low Quality, Worst Quality, plastic, fake, disfigured, deformed, blurry, bad anatomy, blurred, watermark, grainy, signature'
        
            img = pipeline(prompt, negative_prompt=negative_prompt).images[0]
        
            img.save('/content/faces/{}/{}.png'.format(emotion, str(j).zfill(4)))
        
            plt.imshow(img)
            plt.show()


Step-4:
### upload generated zip file on Drive
!zip -r faces.zip /content/faces

from google.colab import drive
drive.mount('/content/gdrive')

!scp '/content/faces.zip' 'YOUR_DRIVE-FOLDER-PATH'
