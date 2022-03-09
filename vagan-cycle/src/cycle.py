from importlib.metadata import requires
from pytorch_lightning import LightningModule
import torch
import torch.optim as optim
from torch import autograd
from models.critics import C3DFCN
from models.mask_generators import UNet
import tifffile as tiff
from imageSaver import Saver
import numpy as np
import itertools

class CycleGAN(LightningModule):
    def __init__(
        self,
        opt,
        LAMBDA=10
    ):
        super().__init__()
        self.forward_net_g, self.forward_net_d = self.init_model(opt)
        self.backward_net_g, self.backward_net_d = self.init_model(opt)
        
        self.optimizer_g, self.optimizer_d = self.init_optimizer(opt, self.forward_net_g, self.backward_net_g, self.forward_net_d, self.backward_net_d)

        self.forward_net_g.apply(self.weights_init)
        self.forward_net_d.apply(self.weights_init)
        self.backward_net_g.apply(self.weights_init)
        self.backward_net_d.apply(self.weights_init)

        self.opt = opt
        self.LAMBDA = LAMBDA
        self.LAMBDA_NORM = opt.lambdaNorm

        self.step = 0
        self.trainStep = 0

        self.losses = []
        self.G_mean_losses = []
        self.D_mean_losses = []

        self.cycleLoss = torch.nn.L1Loss()

        self.first = True

    #################
    ##Init funtions##
    #################

    def weights_init(self, m):
        '''
        Initialize cnn weithgs.
        '''
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data = torch.nn.init.kaiming_normal_(m.weight.data, 2)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def init_model(self, opt):
        '''
        Initialize generator and disciminator
        '''
        net_g = UNet(opt.channels_number, opt.num_filters_g)
        net_d = C3DFCN(opt.channels_number, opt.num_filters_d)
        return net_g, net_d


    def init_optimizer(self, opt, net_g_forward, net_g_backward, net_d_forward, net_d_backward):
        '''
        Initialize optimizers
        '''
        optimizer_g = optim.Adam(itertools.chain(net_g_forward.parameters(), net_g_backward.parameters()), lr=opt.learning_rate_g, betas=(
            opt.beta1, 0.9), weight_decay=1e-5)
        optimizer_d = optim.Adam(itertools.chain(net_d_forward.parameters(), net_d_backward.parameters()), lr=opt.learning_rate_g, betas=(
            opt.beta1, 0.9), weight_decay=1e-5)

        return optimizer_g, optimizer_d

    def configure_optimizers(self):
        return (
            {'optimizer':self.optimizer_g, 'frequency': 1},
            {'optimizer':self.optimizer_d, 'frequency': 100},
        )

    #function taken from: https://github.com/Adi-iitd/AI-Art/blob/6969e0a64bdf6a70aa741c29806fc148de93595c/src/CycleGAN/CycleGAN-PL.py#L680
    @staticmethod
    def set_requires_grad(nets, requires_grad = False):

        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """

        if not isinstance(nets, list): nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    #################
    ##Loss funtions##
    #################

    def generatorLoss(self, map, fake, reconstructed_Map, discriminator, LAMBDA):

        loss = discriminator(fake).mean()
        #cycleLoss = torch.linalg.norm(reconstructed_Map + map, ord=1, dim=2).mean()
        
        cycleLoss = self.cycleLoss(map, -reconstructed_Map)

        totalLoss = loss + LAMBDA*cycleLoss

        return totalLoss

    def discriminatorLoss(self, real, fake):
        return real.mean() - fake.mean()

    def calc_gradient_penalty(self, discriminator, real_data, fake_data, LAMBDA):
        '''
        Calculate gradient penalty as in  "Improved Training of Wasserstein GANs"
        https://github.com/caogang/wgan-gp
        '''
        bs, ch, h, w = real_data.shape

        alpha = torch.rand(bs, 1, device=self.device)
        alpha = alpha.expand(bs, int(real_data.nelement()/bs)).contiguous().view(bs, ch, h, w)


        interpolates = torch.tensor(alpha * real_data + ((1 - alpha) * fake_data), device=self.device, requires_grad=True)

        disc_interpolates = discriminator(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def trainStepD(self, inputToBeFaked, real, discriminator, generator, opt, LAMBDA, calc_grad=True):

        err_d_real = discriminator(real)

        # train with sum (anomaly + anomaly map)
        anomaly_map = generator(inputToBeFaked, sigmoid=opt.sigmoid)
        img_sum = inputToBeFaked + anomaly_map
        
        err_d_anomaly_map = discriminator(img_sum)
        cri_loss = self.discriminatorLoss(err_d_real, err_d_anomaly_map)

        if calc_grad:
            cri_loss += self.calc_gradient_penalty(discriminator, inputToBeFaked, img_sum, LAMBDA)

        return cri_loss

    def trainStepG(self, inputImage, net_g_0, net_g_1, discriminator, opt, LAMBDA):

        anomaly_map = net_g_0(inputImage, sigmoid=opt.sigmoid)
        output = anomaly_map + inputImage

        reconstructed_Map =  net_g_1(output, sigmoid=opt.sigmoid)

        gen_loss = self.generatorLoss(anomaly_map, output, reconstructed_Map, discriminator, LAMBDA)

        return gen_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        '''
        Run the trainig algorithm.
        '''

        anomaly, healthy = batch

        if optimizer_idx == 0:
            ############################
            # Update G forward network
            ############################
            #disable discriminator parameters to prevent optimizations
            self.set_requires_grad([self.forward_net_d, self.backward_net_d], requires_grad = False)
            self.set_requires_grad([self.forward_net_g, self.backward_net_g], requires_grad = True)

            #forward generator
            err_g_forward = self.trainStepG(anomaly, self.forward_net_g, self.backward_net_g, self.forward_net_d, self.opt, self.LAMBDA_NORM)
            #backward generator
            err_g_backward = self.trainStepG(healthy, self.backward_net_g,  self.forward_net_g, self.backward_net_d, self.opt, self.LAMBDA_NORM)

            err_g_total = err_g_forward + err_g_backward / 2 #do we need /2?

            return {'loss': err_g_total, 'err_g_forward': err_g_forward.detach(), 'err_g_backward': err_g_backward.detach()}

        elif optimizer_idx == 1:
            ############################
            # Update D backward network
            ############################
            #reactivate discriminator parameters
            self.set_requires_grad([self.forward_net_g, self.backward_net_g], requires_grad = False)
            self.set_requires_grad([self.forward_net_d, self.backward_net_d], requires_grad = True)

            #forward discriminator
            err_d_forward  = self.trainStepD(anomaly, healthy, self.forward_net_d, self.forward_net_g, self.opt, self.LAMBDA)
            #backward discriminator
            err_d_backward = self.trainStepD(healthy, anomaly, self.backward_net_d, self.backward_net_g, self.opt, self.LAMBDA)

            err_d_total = err_d_forward + err_d_backward / 2 #do we need /2?

            self.trainStep += 1

            return {'loss': err_d_total, 'err_d_forward': err_d_forward.detach(), 'err_d_backward': err_d_backward.detach()}

    def training_epoch_end(self, outputs):
        
        self.step += 1

        avg_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 2 for i in range(2)])
        g_mean_loss = sum([torch.stack([x['loss'] for x in outputs[0]]).mean().item()])
        d_mean_loss = sum([torch.stack([x['loss'] for x in outputs[1]]).mean().item()])
        g_mean_loss_forward = sum([torch.stack([x['err_g_forward'] for x in outputs[0]]).mean().item()])
        g_mean_loss_backward = sum([torch.stack([x['err_g_forward'] for x in outputs[0]]).mean().item()])
        d_mean_loss_forward = sum([torch.stack([x['err_d_forward'] for x in outputs[1]]).mean().item()])
        d_mean_loss_backward = sum([torch.stack([x['err_d_backward'] for x in outputs[1]]).mean().item()])

        self.log('train/total_loss', avg_loss, on_epoch=True)
        self.log('train/g_mean_loss', g_mean_loss, on_epoch=True)
        self.log('train/d_mean_loss', d_mean_loss, on_epoch=True)
        self.log('train/g_loss_forward', g_mean_loss_forward, on_epoch=True)
        self.log('train/g_loss_backward', g_mean_loss_backward, on_epoch=True)
        self.log('train/d_loss_forward', d_mean_loss_forward, on_epoch=True)
        self.log('train/d_loss_backward', d_mean_loss_backward, on_epoch=True)

        self.losses.append(avg_loss)
        self.G_mean_losses.append(g_mean_loss)
        self.D_mean_losses.append(d_mean_loss)

        #change frequency after 25 epochs and in each 100 epoch
        if self.step == 25:
            self.trainer.optimizer_frequencies = [1,5]
        if self.step % 99 == 0:
            self.trainer.optimizer_frequencies = [1, 100]
        if self.step % 100 == 0:
            self.trainer.optimizer_frequencies = [1,5]

        #visualise the networks on full images
        if self.opt.oneImageForward and self.opt.oneImageBackward and self.step % 25 == 0:
            if self.opt.torch:
                oneImage = torch.load(self.opt.oneImageForward, device=self.device)
            else:
                oneImage = torch.from_numpy(tiff.imread(self.opt.oneImageForward)).to(device=self.device)

            oneImage = torch.reshape(oneImage, (1, self.opt.channels_number, np.shape(oneImage)[-2], np.shape(oneImage)[-1]))
            oneImageMap = self.forward_net_g(oneImage, sigmoid=self.opt.sigmoid)

            #we also want to visualize the sanity check
            sanityMap = self.backward_net_g(oneImage + oneImageMap, sigmoid=self.opt.sigmoid)
            oneImageSanity = oneImageMap + sanityMap
            
            #save these tensors on to the server
            Saver.saveAsPng(oneImage.cpu().detach().numpy(), oneImageMap.cpu().detach().numpy(), self.opt.experiment + '/images', self.step, 'forward', self.first)
            Saver.saveAsTiff(oneImage.cpu().detach().numpy(), oneImageMap.cpu().detach().numpy(), self.opt.experiment + '/images', self.step, 'forward', self.first)
            Saver.saveSanity(oneImageSanity.cpu().detach().numpy(), self.opt.experiment + '/images', self.step, 'forward')
            
            if self.opt.torch:
                oneImage = torch.load(self.opt.oneImageBackward, device=self.device)
            else:
                oneImage = torch.from_numpy(tiff.imread(self.opt.oneImageBackward)).to(device=self.device)

            oneImage = torch.reshape(oneImage, (1, self.opt.channels_number, np.shape(oneImage)[-2], np.shape(oneImage)[-1]))
            oneImageMap = self.backward_net_g(oneImage, sigmoid=self.opt.sigmoid)

            #we also want to visualize the sanity check
            sanityMap = self.forward_net_g(oneImage + oneImageMap, sigmoid=self.opt.sigmoid)
            oneImageSanity = oneImageMap + sanityMap

            #save these tensors on to the server
            Saver.saveAsPng(oneImage.cpu().detach().numpy(), oneImageMap.cpu().detach().numpy(), self.opt.experiment + '/images', self.step, 'backward', self.first)
            Saver.saveAsTiff(oneImage.cpu().detach().numpy(), oneImageMap.cpu().detach().numpy(), self.opt.experiment + '/images', self.step, 'backward', self.first)
            Saver.saveSanity(oneImageSanity.cpu().detach().numpy(), self.opt.experiment + '/images', self.step, 'backward')

            if self.first:
                self.first = False

        return None

    def validation_step(self, batch, batch_idx):
        
        anomaly, healthy = batch

        #forward generator
        err_g_forward = self.trainStepG(anomaly, self.forward_net_g, self.backward_net_g, self.forward_net_d, self.opt, self.LAMBDA)
        #backward generator
        err_g_backward = self.trainStepG(healthy, self.backward_net_g,  self.forward_net_g, self.backward_net_d, self.opt, self.LAMBDA)

        total_g_error = err_g_forward + err_g_backward / 2

        #forward discriminator
        err_d_forward  = self.trainStepD(anomaly, healthy, self.forward_net_d, self.forward_net_g, self.opt, self.LAMBDA, calc_grad=False)
        #backward discriminator
        err_d_backward = self.trainStepD(healthy, anomaly, self.backward_net_d, self.backward_net_g, self.opt, self.LAMBDA, calc_grad=False)

        total_d_error = err_d_forward + err_d_backward / 2

        val_avg_loss = total_d_error + total_g_error / 2

        self.log('val/total_loss', val_avg_loss, on_epoch=True)
        self.log('val/g_mean_loss', total_g_error, on_epoch=True)
        self.log('val/d_mean_loss', total_d_error, on_epoch=True)
        self.log('val/g_loss_forward', err_g_forward, on_epoch=True)
        self.log('val/g_loss_backward', err_g_backward, on_epoch=True)
        self.log('val/d_loss_forward', err_d_forward, on_epoch=True)
        self.log('val/d_loss_backward', err_d_backward, on_epoch=True)