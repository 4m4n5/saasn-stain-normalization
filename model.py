# +
import itertools
import functools

import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# YET TO IMPLEMNET
# import utils

from arch import *
from torch.optim import lr_scheduler


# -

class cycleGAN(object):
    def __init__(self, args):
        
        # Set up both gens and discs
        self.Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, 
                              norm=args.norm, use_dropout=args.use_dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, 
                              norm=args.norm, use_dropout=args.use_dropout, gpu_ids=args.gpu_ids)
        
        self.Da = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        
        utils.print_networks([self.Gab,self.Gba,self.Da,self.Db], ['Gab','Gba','Da','Db'])
        
        # Loss functions
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(),self.Gba.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(),self.Db.parameters()), lr=args.lr, betas=(0.5, 0.999))
        

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        
        # Checkpoints
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0
            
    
    def train(self, args):
        # Image transforms
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((args.load_height, args.load_width)),
             transforms.RandomCrop((args.crop_height, args.crop_width)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        dataset_dirs = utils.get_traindata_link(args.dataset_dir)
        
        # Initialize dataloader
        a_loader = torch.utils.data.DataLoader(dsets.ImageFolder(dataset_dirs['trainA'], transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)
        b_loader = torch.utils.data.DataLoader(dsets.ImageFolder(dataset_dirs['trainB'], transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        
        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()
        
        
        for epoch in range(self.start_epoch, args.epochs):
            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)
            
            for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
                # Identify step
                step = epoch * min(len(a_loader), len(b_loader)) + i + 1
                
                
                # Generators ===============================================================
                # Turning off grads for discriminators
                set_grad([self.Da, self.Db], False)
                
                # Zero out grads of the generator
                self.g_optimizer.zero_grad()
                
                # Real images from sets A and B
                a_real = Variable(a_real[0])
                b_real = Variable(b_real[0])
                a_real, b_real = utils.cuda([a_real, b_real])
                
                # Passing through generators
                # Nomenclature. a_fake is fake image generated from b_real in the domain A.
                # NOTE: Gab generate a from b and vice versa
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)
                
                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)
                
                # Both generators should be able to generate the image in its own domain 
                # give an input from its own domain
                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)
                
                # Identity loss
                a_idt_loss = self.L1(a_idt, a_real) * args.lamda * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.lamda * args.idt_coef
                
                # Adverserial loss
                # Da return 1 for an image in domain A
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self_Db(b_fake)
                
                # Label expected here is 1 to fool the discriminator
                expected_label = utils.cuda(Variable(torch.ones(a_fake_dis.size())))
                
                a_gen_loss = self.MSE(a_fake_dis, expected_label)
                b_gen_loss = self.MSE(b_fake_dis, expected_label)
                
                # Cycle Consistency loss
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda
                
                # Total Generator Loss
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss
                
                # Update Generators
                gen_loss.backward()
                self.g_optimizer.step()
                
                # Discriminators ===========================================================
                # Turn on grads for discriminators
                set_grad([self.Da, self.Db], True)
                self.d_optimizer.zero_grad()
                
                # Sample from previously generated fake images
                a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = utils.cuda([a_fake, b_fake])
                
                # Pass through discriminators
                # Discriminator for domain A
                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(a_fake)
                
                # Discriminator for domain B
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(b_fake)
                
                # Expected label for real image is 1
                exp_real_label = utils.cuda(Variable(torch.ones(a_real_dis.size())))
                exp_fake_label = utils.cuda(Variable(torch.zeros(a_fake_dis.size())))
                
                # Discriminator losses
                a_real_dis_loss = self.MSE(a_real_dis, exp_real_label)
                a_fake_dis_loss = self.MSE(a_fake_dis, exp_fake_label)
                b_real_dis_loss = self.MSE(b_real_dis, exp_real_label)
                b_fake_dis_loss = self.MSE(b_real_dis, exp_fake_label)
                
                # Total discriminator loss
                a_dis_loss = (a_fake_dis_loss + a_real_dis_loss)/2
                b_dis_loss = (b_fake_dis_loss + b_real_dis_loss)/2
                
                # Update discriminators
                a_dis_loss.backward()
                b_dis_loss.backward()
                
                self.d_optimizer.step()
                
                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" % 
                      (epoch, i + 1, min(len(a_loader), len(b_loader)), 
                       gen_loss,a_dis_loss+b_dis_loss)
                     )
            
            # Overwrite checkpoint
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()
                                  },
                                  '%s/latest.ckpt' % (args.checkpoint_dir)
                                 )

            # Update learning rates
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()


