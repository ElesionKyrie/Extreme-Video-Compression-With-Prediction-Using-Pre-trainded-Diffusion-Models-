## Please change the directories below to your own
export user="workstation/bohnsix"

export project_dir="mcvd-pytorch"
export code_folder="/home/${user}/${project_dir}" # code folders
export logs_folder="/home/${user}/${project_dir}/Output/Extra/logs" # where to output logs
export data_folder="/home/${user}/scratch/datasets"
export exp_folder=${code_folder}/checkpoints

export dir="${code_folder}"
cd ${dir}

#############
## UCF-101 ##
#############

export config="ucf101"
export data="${data_folder}"
export devices="0, 1"
export nfp="16"

# export exp="ucf10132_big288_4c4_unetm"
# export exp=${exp_folder}/${exp}
# export ckpt="1050000"
# export config_mod="model.ngf=288 model.n_head_channels=288 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32  sampling.batch_size=60 sampling.max_data_iter=1000 model.arch=unetmore"
# sh ./example_scripts/final/base_1f_vidgen_short.sh
export exp="ucf10132_big288_4c4_pmask50_unetm"
export exp=${exp_folder}/${exp}
export ckpt="900000"
export config_mod="data.prob_mask_cond=0.50 model.ngf=288 model.n_head_channels=288 data.num_frames=4 data.num_frames_cond=4 training.batch_size=1  sampling.batch_size=1 sampling.max_data_iter=1000 model.arch=unetmore"
sh ./example_scripts/final/base_1f_vidgen_short.sh