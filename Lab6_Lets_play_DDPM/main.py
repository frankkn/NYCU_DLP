from evaluator import evaluation_model
from dataloader import iclevrLoader
import argparse
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import UNet2DModel
import random
import os
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
import torchvision
import warnings
from tqdm import tqdm


class Trainer:
    def __init__(self, args, model, optimizer, accelerator):
        self.args = args
        self.model = model
        self.train_loader = None
        self.test_loader = data.DataLoader(iclevrLoader(root="./dataset/", mode="test"), batch_size=self.args.test_batch, shuffle=False)
        self.new_test_loader = data.DataLoader(iclevrLoader(root="./dataset/", mode="new_test"), batch_size=self.args.test_batch, shuffle=False)

        self.optimizer = optimizer
        self.lr_scheduler = None 
        self.accelerator = accelerator

        self.timestep = 1000
        self.beta = torch.linspace(1e-4, .002, self.timestep)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0) # 前t個 timestep 的 alpha 累積乘積
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_oneminus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)

        self.one_minus_alpha_cumprod_t_minus_1 =  torch.cat((torch.tensor(1).unsqueeze(0), (1 - self.alpha_cumprod)[:-1]))
        self.one_minus_alpha_cumprod = (1 - self.alpha_cumprod)
        self.sqrt_variance =  torch.sqrt((self.beta * (self.one_minus_alpha_cumprod_t_minus_1 / self.one_minus_alpha_cumprod)))

        self.testing_desc = "Testing"
    
    def train_epoch(self, epoch):
        for epoch in range(1, self.args.epochs + 1):
            self.train_loader = self.train_dataloader(self.args, epoch)

            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=len(self.train_loader) * 500,
            )

            self.train_loader, self.lr_scheduler = self.accelerator.prepare(self.train_loader, self.lr_scheduler)

            self.train(epoch)
            if self.args.save_model and (epoch % 1)==0 :
                self.model.save_pretrained("./model/Unet_" + "epoch_" + str(epoch), variant="non_ema")
                warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated")
                warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None` for 'weights' are deprecated")
                
                self.sample(self.model, self.args.device, self.test_loader, self.args, f"./log/epoch={str(epoch)}/test_epoch={str(epoch)}")
                self.sample(self.model, self.args.device, self.new_test_loader, self.args, f"./log/epoch={str(epoch)}/new_test_epoch={str(epoch)}")


    def train(self, epoch):
        self.model.train()          
        device = self.args.device
        pbar = tqdm(self.train_loader, ncols=120, desc=f"Training Epoch {epoch}")
        for _, (data, cond) in enumerate(pbar):
            data, cond = data.to(device, dtype=torch.float32), cond.to(device)
            cond = cond.squeeze()
            self.optimizer.zero_grad()
            # select t
            rand_t = torch.tensor([random.randint(1, self.timestep) for i in range(data.shape[0])])
            # select noise
            noise = torch.randn(data.shape[0], 3, 64, 64)
            xt = self.compute_xt(data, rand_t, noise)

            ''' 
            Model usage
            # sample: FloatTensor
            # timestep: typing.Union[torch.Tensor, float, int]
            # class_labels: typing.Optional[torch.Tensor] = None
            # return_dict: bool = True 
            # output shape = (batch_size, num_channels, height, width))
            '''

            output = self.model(sample=xt.to(device), timestep=rand_t.to(device), class_labels=cond.to(torch.float32).to(device))
            loss = nn.MSELoss()(output.sample.to(device), noise.to(device))
            self.accelerator.backward(loss)
            self.lr_scheduler.step()
            self.optimizer.step()

            pbar.set_postfix(loss=loss.item(), lr=self.lr_scheduler.get_last_lr()[0])

    def compute_xt(self, data, rand_t, noise):
        # caculate coefficient
        coef_x0 = []
        coef_noise = []
        # select coefficient
        for i in range(data.shape[0]):
            coef_x0.append(self.sqrt_alpha_cumprod[rand_t[i]-1])
            coef_noise.append(self.sqrt_oneminus_alpha_cumprod[rand_t[i]-1])

        coef_x0 = torch.tensor(coef_x0)
        coef_noise = torch.tensor(coef_noise)

        coef_x0 = coef_x0[:, None, None, None]
        coef_noise = coef_noise[:, None, None, None]

        device = self.args.device

        return coef_noise.to(device) * noise.to(device) + coef_x0.to(device) * data.to(device)

    def compute_prev_x(self, xt, t, pred_noise, args):
        coef = 1/torch.sqrt(self.alpha[t-1])
        noise_coef = self.beta[t-1] / self.sqrt_oneminus_alpha_cumprod[t-1]
        if t <= 1 :
            z = 0
        else:
            z = torch.randn(self.args.test_batch, 3, 64, 64)
        sqrt_var = self.sqrt_variance[t-1] 
        mean = coef * (xt - noise_coef * pred_noise)
        prev_x = mean.to("cpu") + sqrt_var.to("cpu") * z
        return prev_x

    def sample(self, model, device, test_loader, args, filename):
        # denormalize
        transform=transforms.Compose([
                transforms.Normalize((0, 0, 0), (1/0.5, 1/0.5, 1/0.5)),
                transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
            ])
        model.eval()
        xt = torch.randn(args.test_batch, 3, 64, 64)
        with torch.no_grad():
            # pbar = tqdm(self.test_loader, ncols=120, desc="Testing")
            for _, (img, cond) in enumerate(test_loader):
                cond = cond.to(device)
                cond = cond.squeeze()
                for t in range(self.timestep, 0, -1):
                    # pred noise
                    output = model(sample = xt.to(args.device), timestep = t, class_labels = cond.to(torch.float32).to(args.device))
                    # compute xt-1
                    xt = self.compute_prev_x(xt.to(args.device), t, output.sample.to(args.device), args)

                # evaluate
                evaluate = evaluation_model()
                acc = evaluate.eval(xt.to(args.device), cond.to(args.device), filename)

                # torch.save(xt, f = filename+".pt")
                print(f"Testing acc: {acc*100:.2f}%")

                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(f'{filename}.txt', 'a') as test_record:
                    test_record.write((f'Average Accuracy : {acc*100:.2f}%\n'))

                # denormalize
                img = transform(xt)
                self.save_images(img, name=filename)

    def save_images(self, images, name):
        grid = torchvision.utils.make_grid(images)
        save_image(grid, fp = name + ".png")

    def train_dataloader(self, args, epoch):
        dataset = iclevrLoader(root="./dataset/", mode="train", partial=args.fast_partial if args.fast_train else args.partial)

        if epoch > args.fast_train_epoch:
            args.fast_train = False

        train_loader = data.DataLoader(dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers)
        return train_loader
        


def main():
    parser = argparse.ArgumentParser(description='Diffusion_Pytorch_Model')
    parser.add_argument('-d', '--device',                            default='cuda')
    parser.add_argument('--train_batch',        type=int,            default=40)
    parser.add_argument('--test_batch',         type=int,            default=32)
    parser.add_argument('--epochs',             type=int,            default=200)
    parser.add_argument('--lr',                 type=float,          default=1e-4 * 0.5)
    parser.add_argument('--gamma',              type=float,          default=0.7)
    parser.add_argument('--save-model',         action='store_true', default=True)

    parser.add_argument('--num_workers',        type=int,            default=14)
    parser.add_argument('--partial',            type=float,          default=1.0,     help="Part of the training dataset to be trained")
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float,          default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int,            default=10,      help="Number of epoch to use fast train mode")

    args = parser.parse_args()
    
    # Training
    # model
    # model = UNet2DModel(sample_size=64, in_channels=3, out_channels=3, layers_per_block=2, class_embed_type=None, block_out_channels=(128, 128, 256, 256, 512, 512),
    #                     down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    #                     up_block_types = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    #                     ).to(args.device)
    # model.class_embedding = nn.Linear(24, 512)
    # model = model.to(args.device)

    # # optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # # Accelerator
    # accelerator = Accelerator()
    # model, optimizer = accelerator.prepare(model, optimizer)

    # trainer = Trainer(args, model, optimizer, accelerator)
    # trainer.train_epoch(args.epochs)


    # DEMO: load model and generate images
    model = UNet2DModel.from_pretrained(pretrained_model_name_or_path ="./model/Unet_epoch_200", variant="non_ema", from_tf=True, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    model.class_embedding = nn.Linear(24 ,512)
    state_dict = torch.load("./model/UNet_epoch_200/diffusion_pytorch_model.non_ema.bin")
    filtered_state_dict = {k[16:]: v for k, v in state_dict.items() if k =="class_embedding.weight" or k=="class_embedding.bias"}
    model.class_embedding.load_state_dict(filtered_state_dict)
    model = model.to(args.device)
    test_loader = data.DataLoader(iclevrLoader(root="./dataset/", mode="test"), batch_size=args.test_batch, shuffle=False)
    new_test_loader = data.DataLoader(iclevrLoader(root="./dataset/", mode="new_test"), batch_size=args.test_batch, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    accelerator = Accelerator()
    model, optimizer = accelerator.prepare(model, optimizer)

    tester = Trainer(args, model, optimizer, accelerator)
    tester.sample(model, args.device, test_loader, args, "./demo/test/test_200")
    tester.sample(model, args.device, new_test_loader, args, "./demo/new_test/new_test_200")


if __name__ == '__main__':
    main()