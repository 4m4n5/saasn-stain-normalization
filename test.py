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


def test(args, epoch):
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

    utils.print_networks([Gab,Gba], ['Gab','Gba'])


    ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_path))
    Gab.load_state_dict(ckpt['Gab'])
    Gba.load_state_dict(ckpt['Gba'])


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
        # Calculate ssim loss
        gray = kornia.color.RgbToGrayscale()
        m = kornia.losses.SSIM(11, 'mean')
        ba_ssim = m(gray((a_real_test + 1) / 2.0), gray((b_fake_test + 1) / 2.0))
        ab_ssim = m(gray((b_real_test + 1) / 2.0), gray((a_fake_test + 1) / 2.0))

    pic = (torch.cat([a_real_test, b_fake_test, a_recon_test, b_real_test, a_fake_test, b_recon_test], dim=0).data + 1) / 2.0

    if not os.path.isdir(args.results_path):
        os.makedirs(args.results_path)

    torchvision.utils.save_image(pic, args.results_path+'/sample_' + str(epoch) + '_' + str(1 - 2*round(ba_ssim.item(), 4)) + '_' + str(1 - 2*round(ab_ssim.item(), 4)) + '.jpg', nrow=args.batch_size)
