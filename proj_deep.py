import matplotlib.pyplot as plt
from torch import nn, optim
import numpy as np
from torchvision import transforms, utils

from torch.utils.data import TensorDataset, DataLoader
from model_utils import unet_model, Discriminator, weights_init
from torch.utils.tensorboard import SummaryWriter
from data_utils import GrayColorImg, ToTensor ,mod_cifar
from losses import total_varation, custom_loss_binary
import torch

writer = SummaryWriter('./run_finit_iters7100/')
composed = transforms.Compose([ToTensor()])
input_folder = '/mnt/dota/dota/Temp/temp/l/gray_scale.npy'
output = '/mnt/dota/dota/Temp/temp/ab/ab/ab1.npy'
batch_size = 8
data_set = GrayColorImg(path_file_input=input_folder, path_file_out=output,
                        transform=composed)
data_set = mod_cifar(transform=composed)
dataloader = DataLoader(data_set, batch_size=batch_size,
                        shuffle=True, num_workers=1)
netD = Discriminator(ngpu=1).cuda()
netD.apply(weights_init)
tst = unet_model(output_shape=[2, 128, 128]).cuda()
load_model = True
if load_model:
    D_state_dict= torch.load('./model_disc2900')
    Gen_state_dict= torch.load('./model_gen2900')
    tst.load_state_dict(Gen_state_dict)
    netD.load_state_dict(D_state_dict)

loss_mse = torch.nn.MSELoss()
criterion = nn.BCELoss()

real_label = 1.
fake_label = 0.
lr_D = 1e-8
lr_G = 4e-8
beta1 = 0.5
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
optimizerG = optim.Adam(tst.parameters(), lr=lr_G, betas=(beta1, 0.999))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
num_epochs = 4
#
#from torchviz import make_dot
#from torch.autograd import Variable
#
#input = Variable(torch.tensor(torch.ones((1, 2, 128,128)))).cuda()
#y = netD(input)
#
#a= make_dot(y.mean() , params=dict(netD.named_parameters()))
#a.view()
Mse_w = 1e-4
TV_w = 1e-12
a_b_variation_w = 1e-12
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        real_cpu = data['img_ab'].cuda()
        netD.zero_grad()
        # Format batch
        b_size = real_cpu.size(0)
        # Forward pass real batch through D
        output = netD(torch.squeeze(real_cpu)).view(-1)
        label = torch.ones_like(output).cuda()
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        fake = tst(torch.squeeze(data['image_grey'].cuda()) )
        #print(fake)
        #print(real_cpu)
        # Generate fake image batch with G

        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        label = torch.zeros_like(output).cuda()

        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        tst.zero_grad()

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)

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
        optimizerG.step()

        # Output training stats
        if i % 10 == 0:
            writer.add_scalar('D(G(z))',
                            D_G_z1,
                            epoch * len(dataloader) + i)
            writer.add_scalar('D(x)',
                            D_x,
                            epoch * len(dataloader) + i)
            writer.add_scalar('mse loss',
                            mse_loss,
                            epoch * len(dataloader) + i)

            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            print('mse loss: %.4f\ttv_loss: %.4f\tab_var: %.4f\tDisc_loss: %.4f'
                  % (mse_loss, tv_loss, a_b_var_loss, discrim_g_loss))

            output_image = data_set.get_rbg_from_lab(np.squeeze(
                data['image_grey'][:, :, 0, :, :]), fake, n=batch_size)

            plt.figure(1)
            plt.imshow(output_image[1].squeeze() )
            plt.figure(2)

            plt.imshow(np.transpose(np.squeeze(
                (data['image_grey'][1, :, :, :, :])/255 ), (1,2,0) ))
            plt.pause(1)

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            torch.save(tst.state_dict(), './model_gen' + str(iters))
            torch.save(netD.state_dict(), './model_disc' + str(iters))
        iters += 1

