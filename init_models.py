from basicsr.utils.download_util import load_file_from_url
import os
from fastai.learner import load_learner
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def download_model(model_url, model_name):
    model_path = os.path.join('weights', model_name + '.pt')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in model_url:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
    return model_path

# ls
country_model = 'country_model_v7am.pkl'
file_url = ['https://cloud.sanarip.org/index.php/s/TbMfZgdDrbRLe7g/download/country_model_v7am.pk1']
country_model_path = download_model(file_url, country_model)
country_model = load_learner(country_model_path)
model_name = 'RealESRGAN_x4plus'
model_url = ['https://cloud.sanarip.org/index.php/s/2XfFdM4CJt52rmm/download/RealESRGAN_x4plus.pth']
real_esrgan_model_path = download_model(model_url, model_name)

model_RDB =RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path=real_esrgan_model_path,
    dni_weight=None,
    model=model_RDB,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    gpu_id=None)
