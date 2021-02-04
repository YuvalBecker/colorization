import matplotlib.pyplot as plt
from torch import nn, optim
import numpy as np
from torchvision import transforms


from torch.utils.data import DataLoader
from model_utils import unet_model, Discriminator, weights_init
from torch.utils.tensorboard import SummaryWriter
from data_utils import ToTensor ,mod_cifar
from losses import total_varation, custom_loss_binary
import torch


class Gan_train():
    def __init__(self):
        self.optimizerD = None
        self.optimizerG = None

    def data_prepare(self, tensorboard_path: str = './runnow/',
                     batch_size: int = 8):
        self.writer = SummaryWriter(tensorboard_path)
        composed = transforms.Compose([ToTensor()])
        self.batch_size = batch_size
        self.data_set = mod_cifar(transform=composed)
        self.dataloader = DataLoader(self.data_set, batch_size=self.batch_size,
                                shuffle=True, num_workers=1)

    def define_model(self, lr_dis: float = 3e-7, lr_gen: float = 4e-7,
                     pretrained_path: list = None,):
        self.netD = Discriminator(ngpu=1).cuda()
        self.netD.apply(weights_init)
        self.gen = unet_model(output_shape=[2, 64, 64]).cuda()
        beta1 = 0.5
        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr_dis, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.gen.parameters(), lr=lr_gen, betas=(beta1, 0.999))

        if pretrained_path is not None:
            D_state_dict = torch.load(pretrained_path[0])
            Gen_state_dict = torch.load(pretrained_path[0][1])
            self.gen.load_state_dict(Gen_state_dict)
            self.netD.load_state_dict(D_state_dict)

    def train(self, num_epochs: int = 2,
              weight_vec: np.ndarray = [1e-4, 1e-12, 1e-12], save_folder='/mnt/dota/dota/Temp/temp/'):
        Mse_w = weight_vec[0]
        TV_w = weight_vec[1]
        a_b_variation_w = weight_vec[2]
        loss_mse = torch.nn.MSELoss()
        criterion = nn.BCELoss()
        G_losses = []
        D_losses = []
        iters = 0

        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):
                real_cpu = data['img_ab'].cuda()
                self.netD.zero_grad()
                output = self.netD(torch.squeeze(real_cpu)).view(-1)
                label = torch.ones_like(output).cuda()
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()
                fake = self.gen(torch.squeeze(data['image_grey'].cuda()) )

                output = self.netD(fake.detach()).view(-1)
                label = torch.zeros_like(output).cuda()

                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()
                self.gen.zero_grad()
                output = self.netD(fake).view(-1)
                label = torch.ones_like(output).cuda()
                mse_loss = loss_mse(fake, torch.squeeze(data['img_ab'].cuda()))
                tv_loss = total_varation(fake)
                a_b_var_loss = custom_loss_binary(fake[:, 0, :, :], fake[:, 1, :, :])
                discrim_g_loss = criterion(output, label)
                errG = discrim_g_loss + torch.clamp(
                    Mse_w * mse_loss, min=0, max=100) + torch.clamp(
                    TV_w * tv_loss , min=0, max=10) + torch.clamp(
                    a_b_variation_w * a_b_var_loss, min=0, max=10)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 10 == 0:
                    self.writer.add_scalar('D(G(z))',
                                    D_G_z1,
                                    epoch * len(self.dataloader) + i)
                    self.writer.add_scalar('D(x)',
                                    D_x,
                                    epoch * len(self.dataloader) + i)
                    self.writer.add_scalar('mse loss',
                                    mse_loss,
                                    epoch * len(self.dataloader) + i)

                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(self.dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    print('mse loss: %.4f\ttv_loss: %.4f\tab_var: %.4f\tDisc_loss: %.4f'
                          % (mse_loss, tv_loss, a_b_var_loss, discrim_g_loss))

                    output_image = self.data_set.get_rbg_from_lab(np.squeeze(
                        data['image_grey'][:, :, 0, :, :]), fake, n=self.batch_size)

                    plt.figure(1)
                    plt.imshow(output_image[1].squeeze() )
                    plt.figure(2)

                    plt.imshow(np.transpose(np.squeeze(
                        (data['image_grey'][1, :, :, :, :])/255), (1, 2, 0)))
                    plt.pause(1)

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(self.dataloader)-1)):
                    torch.save(self.gen.state_dict(), save_folder+'/model_gen' + str(iters))
                    torch.save(self.netD.state_dict(), save_folder+'./model_disc' + str(iters))
                iters += 1


if __name__ == '__main__':
    train_g = Gan_train()
    train_g.data_prepare(tensorboard_path='./runnow/', batch_size = 8)
    train_g.define_model(lr_dis=1e-7,lr_gen=1.5e-7)
    train_g.train(save_folder='/mnt/dota/dota/Temp/temp/')
    #
