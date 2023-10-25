from Network import TestModel

from models.fvd.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
from models.ema import EMAHelper
from models import (ddpm_sampler,
                    ddim_sampler,
                    )

from result_plot import plot
from functools import partial

import torchvision.transforms as Transforms
from function import *

from Inference import inference
import threading
import lpips

import compressai

from compressai.zoo import load_state_dict
import matplotlib
matplotlib.use('Agg')
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
import pandas as pd
import argparse
import datetime
import os
import shutil
import sys
import time
import yaml
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from fvd_utils.calculate_fvd import *





def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument(
        '--config', type=str, default="configs/mine.yml", help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')

    parser.add_argument('--exp', type=str, default='/checkpoints/sender',
                        help='Path for saving running related data.')

    parser.add_argument('--ni', default=True, action='store_true',
                        help="No interaction. Suitable for Slurm Job launcher")

    parser.add_argument('--video_gen', default=True, action='store_true',
                        help='Whether to produce video samples from the conditional model')

    parser.add_argument('-v', '--video_folder', type=str,
                        default='arg_config', help="The folder name of video samples")

    parser.add_argument('--subsample', type=int, default=None,
                        help='# of samples in path, to override config.sampling.subsample, when using sample/test/fast_fid')

    parser.add_argument('--ckpt', type=int, default="900000",
                        help='Model checkpoint # to load from, when using sample/video_gen/test/fast_fid')

    parser.add_argument('--config_mod', nargs='*', type=str,
                        default="model.ngf=192 model.n_head_channels=192")

    parser.add_argument("--data_npy", type=str, default="city_bonn.npy" ,help="data_npy path, shape = B, T, C, H, W ")

    parser.add_argument(
        "--output_path",
        type=str,
        default="/media/myworkstation/文档/test_out/",
        help="result output path",
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        # action="store_true",
        default=True,
        help="enable CUDA",
    )

    parser.add_argument(
        "--plot",
        # action="store_true",
        default=True,
        help="enable plot results",
    )

    parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )

    parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        nargs='+',  # 指定多个值
        default=["NN_CheckPoint/0.pth.tar", "NN_CheckPoint/1.pth.tar", "NN_CheckPoint/2.pth.tar", "NN_CheckPoint/3.pth.tar",
                 "NN_CheckPoint/4.pth.tar", "NN_CheckPoint/5.pth.tar"],
        # required=True,
        help="checkpoint path",
    )

    parser.add_argument(
        "--patch",
        type=int,
        default=64,
        help="padding patch size (default: %(default)s)",
    )

    parser.add_argument("--start_idx", type=int, default=0, help="Start video index")
    parser.add_argument("--end_idx", type=int, default=0, help="End video index")

    args, unknown = parser.parse_known_args()

    args.command = 'python ' + ' '.join(sys.argv)
    args.log_path = os.path.join(args.exp, 'logs')

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Override with config_mod
    if args.config_mod:
        for val in args.config_mod.split(' '):
            val, config_val = val.split('=')
            config_type, config_name = val.split('.')
            try:
                totest = config[config_type][config_name][0]
            except:
                totest = config[config_type][config_name]

            if isinstance(totest, str):
                config[config_type][config_name] = config_val
            else:
                config[config_type][config_name] = eval(config_val)


    if config['model'].get('output_all_frames', False):
        # if False, then wed predict the input-cond frames z, but the z is zero everywhere which is weird and seems irrelevant to predict. So we stick to the noise_in_cond case.
        config['model']['noise_in_cond'] = True

    assert not config['model'].get('cond_emb', False) or (config['model'].get(
        'cond_emb', False) and config['data'].get('prob_mask_cond', 0.0) > 0)

    if config['data'].get('prob_mask_sync', False):
        assert config['data'].get('prob_mask_cond', 0.0) > 0 and config['data'].get(
            'prob_mask_cond', 0.0) == config['data'].get('prob_mask_future', 0.0)



    new_config = dict2namespace(config)


    if args.video_gen:

        new_config.sampling.ckpt_id = args.ckpt or new_config.sampling.ckpt_id
        args.final_only = True
        # if new_config.sampling.final_only:
        os.makedirs(os.path.join(args.exp, 'video_samples'), exist_ok=True)
        args.video_folder = os.path.join(
            args.exp, 'video_samples', args.video_folder)

        if not os.path.exists(args.video_folder):
            os.makedirs(args.video_folder)
        else:
            overwrite = False
            if args.ni:
                overwrite = True
            else:
                response = input(
                    f"Video folder {args.video_folder} already exists.\nOverwrite? (Y/N)")
                if response.upper() == 'Y':
                    overwrite = True

            if overwrite:
                shutil.rmtree(args.video_folder)
                os.makedirs(args.video_folder)
            else:
                print("Output video folder exists. Program halted.")
                sys.exit(0)

        with open(os.path.join(args.video_folder, 'config.yml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        with open(os.path.join(args.video_folder, 'args.yml'), 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)

    # add device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # # # logging.info("Using device: {}".format(device))
    new_config.device = device

    config_uncond = new_config

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config, config_uncond


def my_collate(batch):
    data, _ = zip(*batch)
    data = torch.stack(data).repeat_interleave(preds_per_test, dim=0)
    return data, torch.zeros(len(data))


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X



def get_sampler(version):
    sampler = None
    if version == "DDPM":
        sampler = partial(ddpm_sampler, config=config)
    elif version == "DDIM":
        sampler = partial(ddim_sampler, config=config)
    return sampler


def cal_psnr(img1, img2, maxvalue=1.):
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10((maxvalue ** 2) / mse)

i3d = load_i3d_pretrained(device=torch.device('cuda'))

def calculate_fvd(videos1, videos2, i3d=i3d, device="cuda"):
    # videos [batch_size, timestamps, channel, h, w]
    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    assert videos1.shape == videos2.shape

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    feats1 = get_fvd_feats(videos1, i3d=i3d, device=device)
    feats2 = get_fvd_feats(videos2, i3d=i3d, device=device)
    fvd_results = frechet_distance(feats1, feats2)

    return fvd_results


class SenderCity:

    """
        Input 5 frames and generate 1 frame
        calculate PSNR and update list d
    """

    def __init__(self, threshold, config, args, sampler="DDPM", use_psnr=False, use_lpips=True) -> None:
        self.threshold = threshold
        self.config = config
        self.args = args
        self.T2 = Transforms.Compose([Transforms.Resize((128, 128)), Transforms.ToTensor(),
                                      Transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        self.sampler = get_sampler(sampler)

        self.conditional = self.config.data.num_frames_cond > 0
        self.cond = None
        self.future = getattr(self.config.data, "num_frames_future", 0)
        self.use_psnr = use_psnr
        self.use_lpips = use_lpips
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.config.device)

    def get_model(self):

        ckpt = self.config.sampling.ckpt_id
        ckpt_file = os.path.join(
            self.args.log_path, f'checkpoint_{ckpt}.pt')
        states = torch.load(ckpt_file, map_location=self.config.device)

        from models.better.ncsnpp_more import UNetMore_DDPM
        scorenet = UNetMore_DDPM(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)

        scorenet.load_state_dict(states[0], strict=False)
        scorenet.eval()

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(scorenet)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(scorenet)

        return scorenet.module if hasattr(scorenet, 'module') else scorenet

    def generate_frame(self, input_frames):
        input_frames = data_transform(self.config, input_frames)

        # z
        init_samples_shape = (input_frames.shape[0], self.config.data.channels*self.config.data.num_frames,
                              self.config.data.image_size, self.config.data.image_size)
        init_samples = torch.randn(
            init_samples_shape, device=self.config.device)

        # init_samples

        self.mynet = self.get_model()

        # Generate samples
        gen_samples = self.sampler(init_samples, self.mynet, cond=input_frames, cond_mask=None, n_steps_each=self.config.sampling.n_steps_each, step_lr=self.config.sampling.step_lr, verbose=True, final_only=True, denoise=self.config.sampling.denoise, subsample_steps=getattr(
            self.config.sampling, 'subsample', None), clip_before=getattr(self.config.sampling, 'clip_before', True), t_min=getattr(self.config.sampling, 'init_prev_t', -1), log=True, gamma=getattr(self.config.model, 'gamma', False))

        gen_samples = gen_samples[-1].reshape(gen_samples[-1].shape[0], self.config.data.channels *
                                              self.config.data.num_frames, self.config.data.image_size, self.config.data.image_size)

        pred = gen_samples.to('cpu')
        pred = pred[:, :self.config.data.channels*num_frames_pred]
        pred = inverse_data_transform(self.config, pred.clone())
        pred = pred.unsqueeze(2)

        return pred

    def decide_5to5(self, pred, gt):
        batchsize = gt.shape[0]
        frames_num = gt.shape[1]
        image_size = gt.shape[3:]


        new_d, new_ge = [], []

        for i in range(batchsize):
            for j in range(frames_num):
                _psnr = cal_psnr(pred[i][j], gt[i][j])
                if _psnr >= self.threshold:
                    new_ge.append(pred[i][j])
                    new_d.append(0)
                else:
                    break

        new_d = np.array(new_d).reshape(batchsize, -1)
        new_ge = np.array(new_ge).reshape(
            batchsize, -1, 1, image_size[0], image_size[1])

        return new_d, new_ge

    def decide_5to5_lpips(self, pred, gt):
        pred, gt = pred.to(self.config.device), gt.to(self.config.device)

        batchsize = gt.shape[0]
        frames_num = gt.shape[1]
        image_size = gt.shape[3:]



        new_d, new_ge = [], []

        for i in range(batchsize):
            for j in range(frames_num):
                _lpips = self.loss_fn_alex(
                    pred[i][j].float(), gt[i][j].float()).item()
                print(_lpips, end=" ")

                if _lpips <= self.threshold:

                    new_ge.append(pred[i][j].detach().cpu().numpy())
                    new_d.append(0)

                else:
                    break
        

        new_d = np.array(new_d).reshape(batchsize, -1)
        new_ge = np.array(new_ge).reshape(
            batchsize, -1, 3, image_size[0], image_size[1])

        return new_d, new_ge

    def update(self, x_gt, x_ge, d):
        B, T, C, H, W = x_ge.shape
        idx = x_ge.shape[1]

        print("Generating Frames", str(idx), "to", str(idx+5), "...")

        frames_gt = x_gt[:, idx:idx+5]
        input_frames = x_ge[:, -2:]
        input_frames = input_frames.to(self.config.device)

        input_frames = input_frames.reshape(
            x_ge.shape[0], -1, self.config.data.image_size, self.config.data.image_size)

        pred = self.generate_frame(input_frames).reshape(B, -1, C, H, W)


        if self.use_psnr:
            pred = pred.numpy()
            frames_gt = frames_gt.numpy()
            new_d, new_ge = self.decide_5to5(pred, frames_gt)
        elif self.use_lpips:

            new_d, new_ge = self.decide_5to5_lpips(pred, frames_gt)

        d = np.concatenate((d, new_d), axis=1)

        x_ge = x_ge.detach().cpu().numpy()
        x_ge = np.concatenate((x_ge, new_ge), axis=1)
        x_ge = torch.from_numpy(x_ge)
        return d, x_ge


def compress(model, tensor, patch=576):
    tensor = tensor.to(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)

    if tensor.size(0) == 0:
        img_npy, bits = inference(model, tensor, patch)
        img_npy = img_npy.unsqueeze(0)
        return img_npy, bits
    else:
        npy_list = []
        bits_list = []
        for i in range(tensor.size(0)):
            img_npy, bits = inference(model, tensor[i, :, :, :], patch)
            npy_list.append(img_npy)
            bits_list.append(bits)


        img_con_tensor = torch.cat(npy_list, dim=0)
        img_con_tensor = img_con_tensor.unsqueeze(0)

        return img_con_tensor, bits_list


start_time = time.time()


args, config, config_uncond = parse_args_and_config()

num_frames_pred = config.sampling.num_frames_pred
preds_per_test = getattr(config.sampling, 'preds_per_test', 1)



compressai.set_entropy_coder(args.entropy_coder)



model_cls = TestModel()
model_mapping = {}

for i, path in enumerate(args.paths):
    state_dict = load_state_dict(torch.load(path))
    model = model_cls.from_state_dict(state_dict).eval()
    model_mapping[i] = model


data_gt = torch.from_numpy(np.load(args.data_npy)/255).double()



start_idx = args.start_idx
end_idx = args.end_idx


for databatchidx in range(start_idx, end_idx + 1):

    # inferencing with LPIPS plotting all q values

    all_bpps, all_psnr, all_lpips, all_fvd = [], [], [], []
    output_root = os.path.join(args.output_path, f"output_{databatchidx}")
    # output_dir = os.path.join(output_root, str(databatchidx))
    os.makedirs(output_root, exist_ok=True)  # 如果目录已存在则不会报错

    for q_NN in np.arange(4, 6):
        model = model_mapping[q_NN]
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
        for threshold in np.arange(0.30, 0.02, -0.01):
            threshold = float("%.2f" % threshold)
            print()
            print("*" * 70)
            sender = SenderCity(threshold, config, args)

            print("Runing video batch", databatchidx, "q:", q_NN,
                "threshold LPIPS:", threshold)
            print("Start at", datetime.datetime.now())
            start_sample_time = time.time()



            data_gt_idx = data_gt[databatchidx, :, :, :, :]

            data_condition_org = data_gt_idx[:2, :, :, :]
            data_condition_dec, bits_cond = compress(model, data_condition_org, args.patch)

            x_gt = data_gt_idx.unsqueeze(0)

            x_ge = data_condition_dec

            d = np.array([[1, 1]] * x_gt.shape[0])
            bits_list = []
            bits_list.append(bits_cond)

            while x_ge.shape[1] < 30:
                l = x_ge.shape[1]
                d, x_ge = sender.update(x_gt, x_ge, d)
                if x_ge.shape[1] - l == 0:

                    data_dec, bits_temp = compress(model, data_gt_idx[l:l+2], args.patch)
                    bits_list.append(bits_temp)

                    data_dec = data_dec.detach().cpu().numpy()
                    x_ge = np.concatenate(
                        [x_ge, data_dec], axis=1)

                    x_ge = torch.from_numpy(x_ge)
                    d = np.concatenate(
                        [d, np.array([[1, 1]] * x_gt.shape[0])], axis=1)
            x_ge = x_ge[:, :30]
            d = d[:, :30]

            print(f"d的值为{d}")
            print(f"bpp_list的长度为{len(bits_list)}")
            print(bits_list)

            bits = sum(value for sublist in bits_list for value in sublist)


            NN_bpp = bits / 128 / 128 / 30

            if NN_bpp >= 1.0:
                break

            frames_num = int(30 - np.sum(d))
            psnrlist = [cal_psnr(x_ge[0][i].numpy(), x_gt[0][i].numpy())
                        for i in range(30)]
            lpipslist = [sender.loss_fn_alex(x_gt[0][i].float().cuda(
            ), x_ge[0][i].float().cuda()).item() for i in range(30)]

            print()
            print("average PSNR:%.3f" % np.mean(psnrlist))
            print("we have", int(np.sum(d)), "frames to be transfered in d. ")
            all_psnr.append(psnrlist)
            all_lpips.append(lpipslist)

            ##### calculate FVD
            v1 = x_ge.repeat(2, 1, 1, 1, 1).float()
            v2 = x_gt.repeat(2, 1, 1, 1, 1).float()
            fvd_list = calculate_fvd(v1, v2)
            all_fvd.append(fvd_list)
            all_bpps.append(NN_bpp)

            print()

            print()
            print("d    :    ", list([int(i) for i in d[0]]))
            print()
            print("BPP:           %.5f" % NN_bpp)
            print("FVD:           %f" % fvd_list)
            print("Average PSNR : %.5f" % np.mean(psnrlist))
            print("Average LPIPS: %.5f" % np.mean(lpipslist))

            print()
            print("PNSR :", [float("%.2f" % i) for i in psnrlist])
            print("LPIPS:", [float("%.5f" % i) for i in lpipslist])


            xge = np.concatenate(np.concatenate(
                x_ge.numpy().transpose(0, 1, 3, 4, 2), axis=0), axis=1)
            gt = np.concatenate(np.concatenate(
                x_gt.numpy().transpose(0, 1, 3, 4, 2), axis=0), axis=1)
            t = threading.Thread(target=save_output, args=(
                gt, xge, q_NN, threshold, databatchidx, output_root))
            t.start()

    if args.plot:
        psnr_arr, lpips_arr, fvd_arr = process_data_and_save(databatchidx, all_bpps, all_psnr, all_lpips, all_fvd, output_root)


for databatchidx in range(start_idx, end_idx + 1):
    output_root = os.path.join(args.output_path, f"output_{databatchidx}")

    psnr_arr_plot = np.load(os.path.join(output_root, f'psnr_{databatchidx}.npy'))
    lpips_arr_plot = np.load(os.path.join(output_root, f'lpips_{databatchidx}.npy'))
    fvd_arr_plot = np.load(os.path.join(output_root, f'fvd_{databatchidx}.npy'))

    plot(databatchidx, psnr_arr_plot, lpips_arr_plot, fvd_arr_plot, output_root)



