
######################################################################
############ Sampling DDPM-1000, DDPM-100 and DDIM-100 ###############
######################################################################

# For ease of use using sbatch, we recommend doing something like this (example from our clusters):
# sbatch --time=14:0:0 --account=def-bob --ntasks=1 --gres=gpu:v100l:1 --cpus-per-task=8 --mail-user=my_name_is_skrillex@edm.com --mail-type=ALL --mem=32G -o /scratch/${user}/logs/vidgen_ckpt${ckpt}_nfp${nfp}_${exp}_%j.out --export=config="$config",data="$data",exp="$exp",config_mod="$config_mod",devices="$devices",ckpt="$ckpt",nfp="$nfp" base_1f_vidgen_short.sh
# instead of
# python base_1f_vidgen_short.sh

# Warning: DDPM-1000 is super slow, we did not do it for most models, for faster speed use "base_1f_vidgen_short.sh" instead of "base_1f_vidgen.sh"
# Everything here should fit within a single V-100 with 32Gb of RAM
# For base_1f_vidgen.sh and base_1f_vidgen_short.sh, you need these exported bash variables: config, data, devices, exp, config_mod, ckpt, nfp

# your data folder should look like this:
## BAIR_h5 Cityscapes128_h5 KTH64_h5 MNIST UCF101_64.hdf5

## All checkpoints are available here: https://drive.google.com/drive/folders/15pDq2ziTv3n5SlrGhGM0GVqwIZXgebyD?usp=sharing

###############
## Arguments ##
###############

## Please change the directories below to your own
export user="workstation/bohnsix"

export project_dir="mcvd-pytorch"
export code_folder="/home/${user}/${project_dir}" # code folders
export logs_folder="/home/${user}/${project_dir}/Output/Extra/logs" # where to output logs
export data_folder="/home/${user}/scratch/datasets"
export exp_folder=${code_folder}/checkpoints

export dir="${code_folder}"
cd ${dir}

################
## Cityscapes ##
################

export config="cityscapes_big"
export data="${data_folder}/Cityscapes128_h5"
export devices="0"

exp="city32_big192_5c2_unetm_long"
export exp=${exp_folder}/${exp}
ckpt="900000"
config_mod="model.ngf=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=2 training.batch_size=32 sampling.batch_size=35 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen_short.sh

