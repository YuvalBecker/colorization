import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from model_utils import unet_model, Discriminator, weights_init

from data_utils import ToTensor, mod_cifar
import torch


def evaluation(gen_path: str = './model_gen0', dis_path: str = './model_gen0',
               data_loader_test: DataLoader = 'None', batch_size: int = 8):
    tst = unet_model(output_shape=[2, 128, 128])
    netD = Discriminator(ngpu=1)
    netD.apply(weights_init)

    D_state_dict = torch.load(dis_path)
    Gen_state_dict = torch.load(gen_path)
    tst.load_state_dict(Gen_state_dict)
    netD.load_state_dict(D_state_dict)

    Gen_state_dict = torch.load(gen_path)
    tst.load_state_dict(Gen_state_dict)
    tst_eval = tst.eval()
    netD_eval = netD.eval()
    plt.figure(11)
    for i, data in enumerate(data_loader_test, 0):

        fake = tst_eval(torch.squeeze(data['image_grey']) )
        output_image = data_set.get_rbg_from_lab(np.squeeze(
            data['image_grey'][:, :, 0, :, :]), fake, n=batch_size)
        grade_fake = netD_eval(fake)
        grade = np.squeeze(grade_fake.detach().numpy())
        ind_max = np.argmax(grade)
        print(grade[ind_max])
        if grade[ind_max] > 0.1:
            fig, ax = plt.subplots(1, 2)
            print(grade[ind_max] )
            ax[0].imshow(output_image[ind_max].squeeze())
            ax[1].imshow(np.transpose(np.squeeze(
                (data['image_grey'][ind_max, :, :, :, :])/255), (1, 2, 0)))

            plt.savefig('./output_images/'+str(i)+'.jpg', pid=900)
            plt.close(fig)


if __name__ == '__main__':
    composed = transforms.Compose([ToTensor()])
    data_set = mod_cifar(transform=composed, train=False)
    dataloader = DataLoader(data_set, batch_size=8,
                            shuffle=True, num_workers=1)
    evaluation(data_loader_test=dataloader, gen_path='./model_gen500', dis_path='./model_disc500')

