
import os
import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Non-stationary Diffusion for Time Series Forecasting')

# basic config
parser.add_argument('--ii', type=int, default=0)
parser.add_argument('--use_window_normalization', type=bool, default=True)
parser.add_argument('--is_training', type=int, default=1, help='status')

parser.add_argument('--model', type=str, default='DDPM', 
    help='model name, options: [DDPM]')

parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--pretrain_epochs', type=int, default=20, help='train epochs')

parser.add_argument('--sample_times', type=int, default=1)
parser.add_argument('--beta_dist_alpha', type=float, default=-1)  # -1
parser.add_argument('--our_ddpm_clip', type=float, default=100) # 100

# data loader
parser.add_argument('--seq_len', type=int, default=1440, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=168, help='prediction sequence length')


parser.add_argument('--dataset_name', type=str, default='ETTh1')
parser.add_argument('--weather_type', type=str, default='mintemp', help="['rain' 'mintemp' 'maxtemp' 'solar']")

# Transformer datasets: ECL,ETTh1,ETTh2,ETTm1,ETTm2,Exchange,traffic,weather,illness,wind

# Monash datasets:  https://zenodo.org/communities/forecasting/search?page=3&size=20#
# > weather_dataset: 1332/65981 (3010, 65981)
# > sunspot_dataset_without_missing_values: (1, 73924)
# > [half_hourly] elecdemand_dataset: (Electricity Demand (Elecdemand) Dataset) (1, 17520)
# > [daily] saugeenday_dataset (https://zenodo.org/record/4656058#.Y4cTZWhByUk) (1, 23741)
# > wind_4_seconds_dataset: (1, 7397147)
#[not good] > dominick_dataset: 28/393  (115704, 393)
#[not good] > covid_deaths_dataset: (266-num_vars, 212-seq_len)

# depts datasets
# > caiso
# > production
# > caiso_m
# > production_m
# > synthetic
# > system_KS

# Following are for regression. doesnt work in this version.
# "AustraliaRainfall","HouseholdPowerConsumption1","HouseholdPowerConsumption2","BeijingPM25Quality","BeijingPM10Quality","Covid3Month","LiveFuelMoistureContent","FloodModeling1","FloodModeling2","FloodModeling3","AppliancesEnergy","BenzeneConcentration","NewsHeadlineSentiment","NewsTitleSentiment","BIDMC32RR","BIDMC32HR","BIDMC32SpO2","IEEEPPG","PPGDalia"

parser.add_argument('--features', type=str, default='S',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--num_vars', type=int, default=1, help='encoder input size')

parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

# Diffusion Models
parser.add_argument("--T", type=float, default=1., help="sigma end time in network parametrization")
parser.add_argument('--model_channels', type=int, default=256)
parser.add_argument('--nfe', type=int, default=100)


parser.add_argument('--diff_steps', type=int, default=100, help='number of diffusion steps')
parser.add_argument('--UNet_Type', type=str, default='TST', help=['CNN','VIT','TST'])
parser.add_argument('--type_sampler', type=str, default='dpm', help=["none", "dpm"])
parser.add_argument('--parameterization', type=str, default='x_start', help=["noise", "x_start"])


parser.add_argument('--ablation_study_case', type=str, default="none", help="none, mix_1, ar_1, mix_ar_0, w_pred_loss")

# forecasting task

parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')

parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='2,3', help='device ids of multile gpus')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--batch_size', type=int, default=64, help='32 batch size of train input data')  # 32
parser.add_argument('--test_batch_size', type=int, default=32, help='32 batch size of train input data')  # 32

# parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--conditional', type=bool, default=False, help='conditional training or not')

parser.add_argument('--tag', type=str, default='')
    # TST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if args.use_gpu:
    if args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        torch.cuda.set_device(args.gpu)

args.DATAdir = "datasets"
args.data = "custom"
if args.dataset_name == "ECL":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/electricity/')
    args.data_path = 'electricity.csv'
    args.use_valset = True
if args.dataset_name == "ETTh1":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, '')
    args.data_path = 'ETTh1.csv'
    args.data = "ETTh1"
    args.use_valset = True
if args.dataset_name == "ETTh2":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, '')
    args.data_path = 'ETTh2.csv'
    args.data = "ETTh2"
    args.use_valset = True
if args.dataset_name == "ETTm1":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, '')
    args.data_path = 'ETTm1.csv'
    args.data = "ETTm1"
    args.use_valset = True
if args.dataset_name == "ETTm2":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, '')
    args.data_path = 'ETTm2.csv'
    args.data = "ETTm2"
    args.use_valset = True
if args.dataset_name == "Exchange":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/exchange_rate/')
    args.data_path = 'exchange_rate.csv'
    args.use_valset = True
if args.dataset_name == "traffic":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/traffic/')
    args.data_path = 'traffic.csv'
    args.use_valset = True
if args.dataset_name == "weather":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/weather/')
    args.data_path = 'weather.csv'
    args.use_valset = True
if args.dataset_name == "wind":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/wind/')
    args.data_path = 'wind.csv'
    args.use_valset = True
    args.data = "wind"
    args.target = 'wind_power'
if args.dataset_name == "illness":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_autoformer/illness/')
    args.data_path = 'national_illness.csv'
    args.use_valset = True

if args.dataset_name in ["covid_deaths_dataset","sunspot_dataset_without_missing_values","elecdemand_dataset","saugeenday_dataset","wind_4_seconds_dataset","dominick_dataset","weather_dataset"]:
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_monash')
    args.data_path = ''
    args.use_valset = True

if args.dataset_name in ["caiso", "caiso_m"]:
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_depts/caiso/')
    args.data_path = 'caiso_20130101_20210630.csv'
    args.data = args.dataset_name
    args.use_valset = True
if args.dataset_name in ["production", "production_m"]:
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_depts/nordpool/')
    args.data_path = 'production.csv'
    args.data = args.dataset_name
    args.use_valset = True
if args.dataset_name == "synthetic":
    args.synthetic_mode = 'L'  # ['L', 'Q', 'C', 'LT', 'QT', 'CT']
    args.model_id = "{}_{}_{}_{}".format(args.dataset_name, args.synthetic_mode, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_depts/synthetic/')
    args.data_path = ''
    args.data = 'synthetic'
    args.use_valset = True
if args.dataset_name == "system_KS":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_depts/dynamic_KS/')
    args.data_path = ''
    args.data = 'system_KS'
    args.use_valset = True
if args.dataset_name == "solar":
    args.model_id = "{}_{}_{}".format(args.dataset_name, args.seq_len, args.pred_len)
    args.root_path = os.path.join(args.DATAdir, 'prediction/data_depts/solar/')
    args.data_path = 'solar_energy.csv'
    args.data = 'solar'
    args.use_valset = True

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:

    for ii in range(args.itr):

        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dt{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len, 
            ii,
            )

        if args.tag != '':
            setting += '_' + str(args.tag)

        if args.ablation_study_case != "none":
            setting += '_' + str(args.tag)

        exp = Exp(args)

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        if args.model == "D3VAE":
            exp.D3VAE_train(setting)
        else:
            exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting,args, mode="test")

        torch.cuda.empty_cache()
else:
    ii = args.ii

    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dt{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            ii,
            )

    if args.tag != '':
        setting += '_' + str(args.tag)

    exp = Exp(args)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, args, mode="test")
    torch.cuda.empty_cache()

