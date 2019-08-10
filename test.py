import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from arch import define_Gen, define_Dis
import numpy as np


def test(args, epoch):
    gen_a_losses = []
    gen_b_losses = []
    dis_a_losses = []
    dis_b_losses = []
    
    transform = transforms.Compose(
        [transforms.Resize((args.crop_height, args.crop_width)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset_dirs = utils.get_testdata_link(args.dataset_dir)

    a_test_data = dsets.ImageFolder(dataset_dirs['testA'], transform=transform)
    b_test_data = dsets.ImageFolder(dataset_dirs['testB'], transform=transform)


    a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)
    Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)
    
    Da = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, 
                         gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral=args.spectral)
    Db = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, 
                         gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral=args.spectral)



    utils.print_networks([Gab,Gba], ['Gab','Gba'])


    ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_path))
    Gab.load_state_dict(ckpt['Gab'])
    Gba.load_state_dict(ckpt['Gba'])
    Da.load_state_dict(ckpt['Da'])
    Db.load_state_dict(ckpt['Db'])
    
    # Loss functions
    MSE = nn.MSELoss()
    L1 = nn.L1Loss()

    """ run """
    a_real_test = Variable(iter(a_test_loader).next()[0], requires_grad=True)
    b_real_test = Variable(iter(b_test_loader).next()[0], requires_grad=True)
    a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
            

    Gab.eval()
    Gba.eval()

    with torch.no_grad():
        a_fake_test = Gab(b_real_test)
        b_fake_test = Gba(a_real_test)
        a_recon_test = Gab(b_fake_test)
        b_recon_test = Gba(a_fake_test)
        
    # Both generators should be able to generate the image in its own domain 
    # give an input from its own domain
    a_idt = Gab(a_real_test)
    b_idt = Gba(b_real_test)

    # Identity loss
    a_idt_loss = L1(a_idt, a_real_test) * args.lamda * args.idt_coef
    b_idt_loss = L1(b_idt, b_real_test) * args.lamda * args.idt_coef
        
    # Adverserial loss
    # Da return 1 for an image in domain A
    a_fake_dis = Da(a_fake_test)
    b_fake_dis = Db(b_fake_test)
        
    # Label expected here is 1 to fool the discriminator
    expected_label_a = utils.cuda(Variable(torch.ones(a_fake_dis.size())))
    expected_label_b = utils.cuda(Variable(torch.ones(b_fake_dis.size())))

    a_gen_loss = MSE(a_fake_dis, expected_label_a)
    b_gen_loss = MSE(b_fake_dis, expected_label_b)

    # Cycle Consistency loss
    a_cycle_loss = L1(a_recon_test, a_real_test) * args.lamda
    b_cycle_loss = L1(b_recon_test, b_real_test) * args.lamda

    # Document losses
    gen_a_losses.append([a_gen_loss, a_cycle_loss, a_idt_loss])
    gen_b_losses.append([b_gen_loss, b_cycle_loss, b_idt_loss])

    
    a_real_dis = Da(a_real_test)
    a_fake_dis = Da(a_fake_test)

    # Discriminator for domain B
    b_real_dis = Db(b_real_test)
    b_fake_dis = Db(b_fake_test)

    # Expected label for real image is 1
    exp_real_label_a = utils.cuda(Variable(torch.ones(a_real_dis.size())))
    exp_fake_label_a = utils.cuda(Variable(torch.zeros(a_fake_dis.size())))

    exp_real_label_b = utils.cuda(Variable(torch.ones(b_real_dis.size())))
    exp_fake_label_b = utils.cuda(Variable(torch.zeros(b_fake_dis.size())))

    # Discriminator losses
    a_real_dis_loss = MSE(a_real_dis, exp_real_label_a)
    a_fake_dis_loss = MSE(a_fake_dis, exp_fake_label_a)
    b_real_dis_loss = MSE(b_real_dis, exp_real_label_b)
    b_fake_dis_loss = MSE(b_fake_dis, exp_fake_label_b)

    # Document losses
    dis_a_losses.append([a_fake_dis_loss, a_real_dis_loss])
    dis_b_losses.append([b_fake_dis_loss, b_real_dis_loss])
    
    
    pic = (torch.cat([a_real_test, b_fake_test, a_recon_test, b_real_test, a_fake_test, b_recon_test], dim=0).data + 1) / 2.0

    if not os.path.isdir(args.results_path):
        os.makedirs(args.results_path)

    torchvision.utils.save_image(pic, args.results_path+'/sample_' + str(epoch) + '.jpg', nrow=args.batch_size)
    
    return np.asarray(gen_a_losses), np.asarray(gen_b_losses), np.asarray(dis_a_losses), np.asarray(dis_b_losses)


