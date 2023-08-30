import argparse
import copy
import json
import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision

from dataloader import iclevrLoader
from modules import UNet_conditional, EMA
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import setup_logging, plot_images, save_images, label_to_onehot
from evaluator import evaluation_model
from PIL import Image

label_dict = json.load(open('dataset/objects.json'))
test = json.load(open('dataset/test.json'))
new_test = json.load(open('dataset/new_test.json'))


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def training_stage(args):
    setup_logging(args.run_name)
    device = args.device
    # train_loader = data.DataLoader(iclevrLoader(root="./dataset/", mode="train"), batch_size=args.batch_size, shuffle=True, num_workers=16)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(log_dir=f'./logs/{args.run_name}')
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    for epoch in range(args.epochs):
        logging.info(f"Start epoch {epoch}:")
        # train_loader = data.DataLoader(iclevrLoader(root="./dataset/", mode="train"), batch_size=args.batch_size, shuffle=True, num_workers=16)
        train_loader = train_dataloader(args, epoch)
        l = len(train_loader)
        pbar = tqdm(train_loader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.from_numpy(label_to_onehot(test[6], label_dict)).to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            # plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(),os.path.join("models", args.run_name, f"ckpt.pth"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pth"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pth"))

def train_dataloader(args, epoch):

    dataset = iclevrLoader(root="./dataset/", mode="train", \
                           partial=args.fast_partial if args.fast_train else args.partial)

    if epoch > args.fast_train_epoch:
        args.fast_train = False

    train_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return train_loader

def testing_stage(args):
    logging.info(f"Start testing:")

    default_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    device = args.device
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
        x = diffusion.sample(model, 1, new_y[i], cfg_scale=3)
        img_path = os.path.join("results", args.run_name, "new_test", f"new_test_{i}.jpg")
        save_images(x, img_path)
        image = Image.open(img_path).convert('RGB')
        image = default_transforms(image).to(device)
        new_img.append(image.unsqueeze(0))

    new_img = torch.cat(new_img)
    new_y = torch.stack(new_y)
    test = evaluation_model()
    print(f'New test acc: {test.eval(new_img, new_y) * 100 :.2f}')



def main(args):
    if args.train:
        training_stage(args)
    else:
        testing_stage(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',           type=str,            default="DDPM_conditional_three_layers")
    parser.add_argument('--epochs',             type=int,            default=50)
    parser.add_argument('--batch_size',         type=int,            default=18)
    parser.add_argument('--image_size',         type=int,            default=64)
    parser.add_argument('--num_classes',        type=int,            default=24)
    parser.add_argument('--dataset_path',       type=str,            default="./dataset")
    parser.add_argument('--partial',            type=float,          default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--device',             type=str,            default="cuda")
    parser.add_argument('--lr',                 type=float,          default=3e-4)
    parser.add_argument('--train',                                   default=True)
    parser.add_argument('--test_only',          action='store_true')
    parser.add_argument('--num_workers',        type=int,            default=14)

    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,   help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=10,      help="Number of epoch to use fast train mode")


    args = parser.parse_args()
    
    main(args)