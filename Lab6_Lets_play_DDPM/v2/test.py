import torch
from modules import UNet_conditional
from main import Diffusion
from utils import setup_logging, plot_images, save_images, label_to_onehot
import json
import os 
from evaluator import evaluation_model
from PIL import Image
import torchvision
from tqdm import tqdm

default_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

label_dict = json.load(open('dataset/objects.json'))
test = json.load(open('dataset/test.json'))
new_test = json.load(open('dataset/new_test.json'))

exp_name = "DDPM_conditional_three_layers"

device = 'cuda:0'
model = UNet_conditional(num_classes=24).to(device)
ckpt = torch.load('./models/DDPM_conditional_three_layers/ckpt.pth')
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)


new_y = [] # label
new_img = [] # generated img

for i in new_test:
    label = torch.Tensor(label_to_onehot(i, label_dict)).to(device) # i = ["brown cube", "purple cube"]
    new_y.append(label)

for i in range(len(new_y)):
# for i in tqdm(range(len(new_y))):
    x = diffusion.sample(model, 1, new_y[i], cfg_scale=3)
    img_path = os.path.join("results", exp_name, "new_test", f"new_test_{i}.jpg")
    save_images(x, img_path)
    image = Image.open(img_path).convert('RGB')
    image = default_transforms(image).to(device)
    new_img.append(image.unsqueeze(0))

new_y = torch.stack(new_y)
test = evaluation_model()
new_img = torch.cat(new_img)
print(test.eval(new_img, new_y))