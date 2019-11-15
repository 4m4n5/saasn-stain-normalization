# %%
import warnings

# Essentials
import os
import glob
import javabridge
import bioformats
import numpy as np

# Torch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Image functions
from PIL import Image as PILImage

# Own code
import utils
from arch import define_Gen

# DL Prediction
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.callbacks.hooks import *
from fastai.callback import *

# Misc
import matplotlib.cm as mpl_color_map
import copy

warnings.filterwarnings('ignore')


# %%
args = {
    'epochs': 100,
    'decay_epoch': 60,
    'batch_size': 4,
    'lr': 0.0002,
    'load_height': 128,
    'load_width': 128,
    'gpu_ids': '0',
    'crop_height': 128,
    'crop_width': 128,
    'alpha': 6, # Cyc loss
    'beta': 5, # Scyc loss
    'gamma': 2, # Dssim loss 
    'delta': 0.1, # Identity
    'training': True,
    'testing': True,
    'results_dir': '/project/DSone/as3ek/data/ganstain/run2/vsi_svs/results/',
    'dataset_dir': '/project/DSone/as3ek/data/ganstain/run2/vsi_svs/',
    'checkpoint_dir': '/project/DSone/as3ek/data/ganstain/run2/vsi_svs/checkpoint/',
    'norm': 'batch',
    'use_dropout': False,
    'ngf': 64,
    'ndf': 64,
    'gen_net': 'unet_128',
    'dis_net': 'n_layers',
    'self_attn': True,
    'spectral': True,
    'log_freq': 50,
    'custom_tag': 'vsi_svs',
    'gen_samples': True,
    'specific_samples': False
}

args = utils.Arguments(args)
args = utils.process_args(args)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Load and initialize normalization modelimport matplotlib.cm as mpl_color_map
import copy
def define_load_gen(args, one_direction=True, gen_name='Gba'):
    ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_path))
    if one_direction:
        G = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                        use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)
        
        G.load_state_dict(ckpt[gen_name])
        G.eval()
        return G
    else:
        Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                            use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)
        Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= args.use_dropout, gpu_ids=args.gpu_ids, self_attn=args.self_attn, spectral = args.spectral)
        Gab.load_state_dict(ckpt['Gab'])
        Gba.load_state_dict(ckpt['Gba'])
        Gab.eval()
        Gba.eval()
        return Gab, Gba


# %%
# Load and initialize DL prediction model
def data_learner_init(PATH, sz, tfms, normalize_stats, model_load_name):
    data = ImageDataBunch.from_folder(PATH, ds_tfms=tfms, size=sz)
    print('Data Loaded')
    if normalize_stats:
        if normalize_stats == 'batch_stats':
            normalize_stats = data.batch_stats()
        data.normalize(normalize_stats)
        print('Data Normalized')
    
    learn = cnn_learner(data, models.resnet50, metrics=accuracy)
    print('Model Initialized')
    if model_load_name:
        learn.load(model_load_name)
        print('Moel Loaded')
    
    return data, learn, normalize_stats


# %%
# Common paramters
patch_size = 1000
target_size = 256
target = '/scratch/as3ek/misc/feature_gradcam/group_2/' # for Gradcam WSI
thresh = 0 # %-age tissue coverrage cutoff
overlap = 0 # %-age area

# Stain Normalization Parameters
UNNORM_WSI_PATH = '/project/DSone/biopsy_images/SEEM_New_crops/SEEM_New_Crops/'
saasn_transform = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
one_direction = True # If this is false. a -> b -> a will happen. Edit code for otherwise.
gen_name = 'Gba' # Gba to generate b given a, i.e., a -> b

# DL Prediction parameters
DL_DATA_PATH = "/project/DSone/as3ek/data/patches/1000/classification/normal_all__ee_vsi/"
sz = target_size
tfms = get_transforms(do_flip=True, flip_vert=True, max_zoom=1.1)
dl_normalize_stats = ([0.7261, 0.5643, 0.6989], [0.1835, 0.2359, 0.1813]) # ([mean, mean, mean], [std, std, std]) if manual. False is no normalization
model_load_name = 'unfreeze50-epoch-1-meanstdnorm' # False if none
cl = 0 # EE - 0 | Normal - 1

# Gene correlation parameters
custom_list = ['LRAT','SHMT1','MTHFS','SLC19A1','CYB5A','ISX','CYP4F2']
corr_data = pd.read_csv('/scratch/as3ek/histo_visual_recog/scripts/data/corr_metric_cutoff_7.csv')
top_genes = pd.read_csv('/scratch/as3ek/histo_visual_recog/scripts/data/top_gene_feature_list.csv')


# %%
# Load Normalization model
if one_direction:
    G = define_load_gen(args, one_direction, gen_name)    
else:
    Gab, Gba = define_load_gen(args, False, '')
    
# Load DL model
data, learn, dl_norm_stats = data_learner_init(DL_DATA_PATH, sz, tfms, dl_normalize_stats, model_load_name)


# %%
javabridge.start_vm(class_path=bioformats.JARS)


# %%
def get_img_paths_vsi(train_paths):
    images = {}
    files = glob.glob(os.path.join(train_paths, '*.vsi'))
    for fl in files:
        flbase = os.path.basename(fl)
        flbase_noext = os.path.splitext(flbase)[0]
        images[flbase_noext] = fl
    return images


# %%
def _hook_inner(m,i,o): return o if isinstance(o,Tensor) else o if is_listy(o) else list(o)

def hook_output (module:nn.Module, detach:bool=True, grad:bool=False)->Hook:
    "Return a `Hook` that stores activations of `module` in `self.stored`"
    return Hook(module, _hook_inner, detach=detach, is_forward=not grad)


# %%
def gradcam_hm(learn, im, cl, c, ft_lst, heatmap_thresh=16):
    m = learn.model.eval()
    im = Image(im[0])
    cl = int(cl)
    xb,_ = data.one_item(im) #put into a minibatch of batch size = 1

    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cl)].backward() 
    acts  = hook_a.stored[0].cpu() #activation maps
    
    # Get one hot encoding for selected features
    ft_lst_one_hot = np.zeros((len(ft_lst), 2048))
    ft_lst_one_hot[np.arange(len(ft_lst)), ft_lst] = 1
    ft_lst_one_hot = np.sum(ft_lst_one_hot, axis=0)
    ft_lst_one_hot = torch.FloatTensor(ft_lst_one_hot[...,None,None])
    
    if (acts.shape[-1]*acts.shape[-2]) >= heatmap_thresh:
        grad = hook_g.stored[0][0].cpu()
        grad_chan = grad.mean(1).mean(1)
        mult = F.relu(((acts*grad_chan[...,None,None])*ft_lst_one_hot).sum(0))
        
        xb_im = Image(denormalize(xb[0], torch.tensor(dl_norm_stats[0]), torch.tensor(dl_norm_stats[1])))
        
        mult = mult * preds[0,int(cl)]
        return xb_im, mult
    
def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = PILImage.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = PILImage.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = PILImage.new("RGBA", org_im.size)
    heatmap_on_image = PILImage.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = PILImage.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


# %%
files = list(get_img_paths_vsi(UNNORM_WSI_PATH).values())
num_files = len(files)

for i, file_path in enumerate(files):

    gene_list = list(list(np.unique(top_genes['Gene Symbol'])))
    
    if len(custom_list) != 0:
        gene_list = custom_list 
    
    for gene in gene_list:
        feature_list = list(corr_data[corr_data['Gene Symbol'] == gene]['Feature'])
        
        image = bioformats.ImageReader(file_path)
        rescale = target_size / patch_size
        height, width, c = np.array(image.read(rescale=False)).shape
        new_dims = int(rescale * (width // patch_size) * patch_size), int(rescale * (height // patch_size) * patch_size)

        hm_rescale = 8 / target_size
        hm_dims = int(new_dims[0] * hm_rescale), int(new_dims[1] * hm_rescale)
        file = file_path.split('/')[-1]

        # Initialize x and y coord
        x_cord = 0
        y_cord = 0

        if hm_dims == (0, 0):
            continue

        # Full scale wsi
        wsi = PILImage.new('RGB', new_dims)
        com_hm = PILImage.new('L', hm_dims)

        while y_cord + patch_size < height - 0:
            while x_cord + patch_size < width - 0:
                patch = PILImage.fromarray(np.array(image.read(rescale=False, XYWH=(x_cord, y_cord, patch_size, patch_size))))

                patch = patch.convert('RGB')
                patch = patch.resize((target_size, target_size))
                patch = np.array(patch)

                patch = patch.transpose(2, 0, 1)
                patch = patch / 255.
                patch = torch.FloatTensor(patch).to(device)
                patch = saasn_transform(patch)
                patch = patch.unsqueeze(0)

                if one_direction:
                    out = G(patch)
                else:
                    out = Gba(patch)
                    out = Gab(out)

                out = out.detach().cpu()
                out = (out + 1) / 2

                c = file + str(x_cord) + '_' + str(y_cord)
                xb_img, mult = gradcam_hm(learn, out, cl, c, feature_list)

                img = out.numpy()[0]
                img = np.transpose(img, (1,2,0))
                patch_join = PILImage.fromarray(np.uint8(img*255))
                wsi.paste(patch_join, (int(x_cord*rescale), int(y_cord*rescale)))

                hm = mult.detach().cpu().numpy()

                if x_cord == 0:
                    hm_row = hm
                else:
                    hm_row = np.concatenate((hm_row, hm), axis=1)

                # Taking care of overlap
                x_cord = int(x_cord + (1 - overlap) * patch_size)

            # Taking care of overlap
            if y_cord == 0:
                com_hm = hm_row
            else:
                com_hm = np.concatenate((com_hm, hm_row), axis = 0)

            y_cord = int(y_cord + (1 - overlap) * patch_size)
            x_cord = 0

        cam = com_hm
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1

        # Threshold
        cam[cam<0.3] = 0

        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        # Resize requires width, height
        cam = np.uint8(PILImage.fromarray(cam).resize((np.array(wsi).shape[1],
                       np.array(wsi).shape[0]), PILImage.BILINEAR))/255


        no_trans_heatmap, heatmap_on_image = apply_colormap_on_image(wsi, cam, 'hsv')
        heatmap_on_image.save(target + file.split('.')[0] + '_' + gene + '.png')

    print(str(i + 1) + '/' + str(num_files) + ' Complete!')


# %%
file_path

# %%
