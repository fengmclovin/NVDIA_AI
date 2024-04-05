# pip install stylegan2_pytorch

import torch
import torchvision
import torchvision.transforms as transforms
from stylegan2_pytorch import StyleGAN2

# Load the pre-trained StyleGAN2 model
model = StyleGAN2('stylegan2-ffhq-config-f.pt')

# Generate synthetic images
with torch.no_grad():
    noise = torch.randn(1, model.latent_dim).cuda()
    image = model(noise)

# Convert the generated image tensor to a PIL image
transforms.ToPILImage()(image.squeeze(0).cpu()).show()
