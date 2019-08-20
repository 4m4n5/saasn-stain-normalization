import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from arch import define_Gen, define_Dis
import kornia


def gen_samples(args, epoch):
    transform = transforms.Compose(
        [transforms.Resize((args.crop_height, args.crop_width)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset_dirs = utils.get_testdata_link(args.dataset_dir)

    a_test_data = dsets.ImageFolder(dataset_dirs['testA'], transform=transform)
    b_test_data = dsets.ImageFolder(dataset_dirs['testB'], transform=transform)


    a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)
    Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)

    utils.print_networks([Gab,Gba], ['Gab','Gba'])


    ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_path))
    Gab.load_state_dict(ckpt['Gab'])
    Gba.load_state_dict(ckpt['Gba'])

    
    ab_ssims = []
    ba_ssims = []
    a_names = []
    b_names = []
    """ run """
    for i, (a_real_test, b_real_test) in enumerate(zip(a_test_loader, b_test_loader)):
        a_fnames = a_test_loader.dataset.samples[i*16 : i*16 + 16]
        b_fnames = b_test_loader.dataset.samples[i*16 : i*16 + 16]
        
        a_real_test = Variable(a_real_test[0], requires_grad=True)
        b_real_test = Variable(b_real_test[0], requires_grad=True)
        a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])

        gray = kornia.color.RgbToGrayscale()
        m = kornia.losses.SSIM(11, 'mean')
        
        Gab.eval()
        Gba.eval()

        with torch.no_grad():
            a_fake_test = Gab(b_real_test)
            b_fake_test = Gba(a_real_test)
            a_recon_test = Gab(b_fake_test)
            b_recon_test = Gba(a_fake_test)
            # Calculate ssim loss
            
            for j in range(args.batch_size):
                a_real = a_real_test[j].unsqueeze(0)
                b_fake = b_fake_test[j].unsqueeze(0)
                a_recon = a_recon_test[j].unsqueeze(0)
                b_real = b_real_test[j].unsqueeze(0)
                a_fake = a_fake_test[j].unsqueeze(0)
                b_recon = b_recon_test[j].unsqueeze(0)

                ba_ssim = m(gray((a_real + 1) / 2.0), gray((b_fake + 1) / 2.0))
                ab_ssim = m(gray((b_real + 1) / 2.0), gray((a_fake + 1) / 2.0))
                
                ab_ssims.append(ab_ssim)
                ba_ssims.append(ba_ssim)

                pic = (torch.cat([a_real_test, b_fake_test, a_recon_test, b_real_test, a_fake_test, b_recon_test], dim=0).data + 1) / 2.0

                path = args.results_path + '/b_fake/'
                image_path = path + a_fnames[j][0].split('/')[-1]
                if not os.path.isdir(path):
                    os.makedirs(path)
                    torchvision.utils.save_image(b_fake, image_path)
                
                a_names.append(a_fnames[j][0].split('/')[-1])
                    
                path = args.results_path + '/a_recon/'
                image_path = path + a_fnames[j][0].split('/')[-1]
                if not os.path.isdir(path):
                    os.makedirs(path)
                    torchvision.utils.save_image(a_recon, image_path)
                    
                path = args.results_path + '/a_fake/'
                image_path = path + b_fnames[j][0].split('/')[-1]
                if not os.path.isdir(path):
                    os.makedirs(path)
                    torchvision.utils.save_image(a_fake, image_path)
                
                b_names.append(b_fnames[j][0].split('/')[-1])
                    
                path = args.results_path + '/b_recon/'
                image_path = path + b_fnames[j][0].split('/')[-1]
                if not os.path.isdir(path):
                    os.makedirs(path)
                    torchvision.utils.save_image(b_recon, image_path)
        
        df1 = pd.DataFrame(list(zip(a_names, ba_ssims)), columns =['Name', 'SSIM_A_to_B']) 
        df2 = pd.DataFrame(list(zip(b_names, ab_ssims)), columns =['Name', 'SSIM_B_to_A']) 
