from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from networks import NetG, NetD, weights_init
from loss_fcns import l2_loss
from evaluate import evaluate
import utils

import torch.nn.functional as func
class BaseModel():
    def __init__(self, opt, dataloader):
        self.seed(opt.manualseed)
        self.opt = opt
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
    ##

    ##
    def set_input(self, input: torch.Tensor):
        """ Set input and ground truth
        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])


    def seed(self, seed_value):
        """ Seed

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    def get_errors(self):
        """ Get netD and netG errors.
        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_enc', self.err_g_enc.item())])

        return errors

##
    def get_current_images(self):
        """ Returns current images.
        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.
        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pth' % (weight_dir))

    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            # self.optimize()
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)


        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch + 1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.netg.train()
            self.train_one_epoch()
            '''
            res = self.test2()
            if res[self.opt.metric] > best_auc:
                best_auc = res[self.opt.metric]
                self.save_weights(self.epoch)
        print(">> Training model %s.[Done]" % self.name)
        '''
    def test2(self):
        
        self.netg.eval()
        self.opt.phase = 'test'

            # Create big error tensor for the test set.
        self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32,
                                         device=self.device)
        self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,
                                         device=self.device)
        self.latent_i = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.latent_size), dtype=torch.float32,
                                        device=self.device)
        self.latent_o = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.latent_size), dtype=torch.float32,
                                        device=self.device)
        self.times = []
        self.total_steps = 0
        epoch_iter = 0
        for i, data in enumerate(self.dataloader['test'], 0):
            self.total_steps += self.opt.batchsize_test
            epoch_iter += self.opt.batchsize_test
            time_i = time.time()
            self.set_input(data)
                
            self.netg.zero_grad()
            self.fake, latent_i, latent_o = self.netg(self.input)
                
            error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
            #error = func.mse_loss(latent_i,latent_o)
            #error.backward()
            #GradCon Results
                
            recon_loss = func.mse_loss(self.input,self.fake)
            recon_loss.backward()
            self.grad_loss = 0
            for j in range(4):
                target_grad = self.netg.decoder.main[3 * j].weight.grad
                self.grad_loss += -1 * func.cosine_similarity(target_grad.view(-1, 1),self.ref_grad[j].avg.view(-1, 1), dim=0)
                
            self.grad_loss = self.grad_loss/4
                
            time_o = time.time()
            #print('Gradient: {}'.format(self.grad_loss))
            self.an_scores[i * self.opt.batchsize_test: i * self.opt.batchsize_test + error.size(0)] = error.reshape(
                error.size(0)) + self.grad_loss *self.opt.w_grad
            self.gt_labels[i * self.opt.batchsize_test: i * self.opt.batchsize_test + error.size(0)] = self.gt.reshape(
                error.size(0))
            self.latent_i[i * self.opt.batchsize_test: i * self.opt.batchsize_test + error.size(0), :] = latent_i.reshape(
                error.size(0), self.opt.latent_size)
            self.latent_o[i * self.opt.batchsize_test: i * self.opt.batchsize_test + error.size(0), :] = latent_o.reshape(
                error.size(0), self.opt.latent_size)

            self.times.append(time_o - time_i)
    # Measure inference time.
        self.times = np.array(self.times)
        self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
        self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                    torch.max(self.an_scores) - torch.min(self.an_scores))
            # auc, eer = roc(self.gt_labels, self.an_scores)
        auc = evaluate(self.gt_labels.detach().numpy(), self.an_scores.detach().numpy(), metric=self.opt.metric)
        performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (self.opt.metric, auc)])
        print(performance)

        return performance
    def test(self):
        """ Test GANomaly model.
        Args:
            dataloader ([type]): Dataloader for the test set
        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32,
                                         device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,
                                         device=self.device)
            self.latent_i = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.latent_size), dtype=torch.float32,
                                        device=self.device)
            self.latent_o = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.latent_size), dtype=torch.float32,
                                        device=self.device)

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize_test
                epoch_iter += self.opt.batchsize_test
                time_i = time.time()
                self.set_input(data)
                
                self.netg.zero_grad()
                self.fake, latent_i, latent_o = self.netg(self.input)
                
            
           
                #GradCon Results
                
                recon_loss = Variable(func.mse_loss(self.input,self.fake),requires_grad=True)
                recon_loss.backward()
                self.grad_loss = 0
                for j in range(4):
                    target_grad = self.netg.decoder.main[3 * j].weight.grad
                    self.grad_loss += -1 * func.cosine_similarity(target_grad.view(-1, 1),self.ref_grad[j].avg.view(-1, 1), dim=0)
                
                self.grad_loss = self.grad_loss/4
                
                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                time_o = time.time()
                #print('Error: {} Gradient: {}'.format(error,self.grad_loss))
                self.an_scores[i * self.opt.batchsize_test: i * self.opt.batchsize_test + error.size(0)] = error.reshape(
                    error.size(0)) #+ self.grad_loss *self.opt.w_grad
                self.gt_labels[i * self.opt.batchsize_test: i * self.opt.batchsize_test + error.size(0)] = self.gt.reshape(
                    error.size(0))
                self.latent_i[i * self.opt.batchsize_test: i * self.opt.batchsize_test + error.size(0), :] = latent_i.reshape(
                    error.size(0), self.opt.latent_size)
                self.latent_o[i * self.opt.batchsize_test: i * self.opt.batchsize_test + error.size(0), :] = latent_o.reshape(
                    error.size(0), self.opt.latent_size)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()


            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                        torch.max(self.an_scores) - torch.min(self.an_scores))
            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (self.opt.metric, auc)])
            print(performance)

            return performance
        

##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, opt, dataloader):
        super(Ganomaly, self).__init__(opt, dataloader)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0
        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)
        # Initialize Gradient Management
        self.ref_grad = []
        for i in range(4):
            layer_grad = utils.AverageMeter()
            layer_grad.avg = torch.zeros(self.netg.decoder.main[3 * i].weight.shape).to(self.device)
            self.ref_grad.append(layer_grad)
        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, self.opt.num_channels, self.opt.im_size, self.opt.im_size), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, self.opt.num_channels, self.opt.im_size, self.opt.im_size), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

    ##
    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.l_adv(self.netd(self.input)[1], self.netd(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, self.input)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.grad_loss = 0

        # Gradient Computation
        recon_loss = func.mse_loss(self.input,self.fake)
        for i in range(4):
            wrt = self.netg.decoder.main[3 * i].weight
            target_grad = torch.autograd.grad(recon_loss, wrt, create_graph=True, retain_graph=True)[0]
            self.grad_loss += -1 * func.cosine_similarity(target_grad.view(-1, 1),
                                                     self.ref_grad[i].avg.view(-1, 1), dim=0)

        if self.ref_grad[0].count == 0:
            self.grad_loss = torch.FloatTensor([0.0]).to(self.device)
        else:
            self.grad_loss = self.grad_loss / 4

        self.err_g = self.err_g_adv * self.opt.w_adv + \
                     self.err_g_con * self.opt.w_con + \
                     self.err_g_enc * self.opt.w_enc + self.grad_loss * self.opt.w_grad
         
        self.err_g.backward(retain_graph=True)
        # Update the reference gradient
        self.ref_grad_curr=[]
        for i in range(4):
            self.ref_grad[i].update(self.netg.decoder.main[3 * i].weight.grad, 1)
            self.ref_grad_curr.append(self.ref_grad[i])
            

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('   Reloading net d')

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()