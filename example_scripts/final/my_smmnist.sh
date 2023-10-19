# Please change the directories below to your own
export user="workstation/bohnsix"

export project_dir="mcvd-pytorch"
export code_folder="/home/${user}/${project_dir}" # code folders
export logs_folder="/home/${user}/${project_dir}/Output/Extra/logs" # where to output logs
export data_folder="/media/workstation/hdd2/mmnist/MNIST"
export exp_folder=${code_folder}/checkpoints

export dir="${code_folder}"
cd ${dir}

#############
## SMMNIST ##
#############

export config="sender"
export data="${data_folder}"
export devices="0, 1"
export nfp="5"

export exp="sender"
export exp=${exp_folder}/${exp}  # 
export ckpt="700000"
export config_mod="sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2"
sh ./example_scripts/final/base_1f_vidgen.sh

# export exp="SMMNIST_big_c5t5_SPADE"
# export exp=${exp_folder}/${exp}
# export ckpt="140000"
# export config_mod="model.spade=True sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2"
# sh ./example_scripts/final/base_1f_vidgen.sh
