REM @echo off 
setLocal EnableDelayedExpansion

SET lr_step0=2.5e-4
SET lr_stepN=2.5e-5
SET decay=5e-4

SET /A epochs=6
SET /A eval=1

SET source_datasets=gta
SET target_datasets=cityscapes
SET /A batch_size=1
SET /A target_batch_size=1

SET names=Final
SET methods=FT
SET task=6-6-8
SET steps=0

SET lmsq=0.1
SET ratio=0.2
SET lkd=2

SET backup=500

SET logdir=logs/
SET out_file=outputs/output_%task%_%names%_step%steps%.txt
echo %out_file%

CALL python -u -m torch.distributed.launch 1> %out_file% 2>&1 ^
--nproc_per_node=1 run_uda.py ^
--batch_size %batch_size% ^
--logdir %logdir% ^
--dataset %source_datasets% ^
--weight_decay %decay% ^
--name %names% ^
--task %task% ^
--step %steps% ^
--lr %lr_step0% ^
--epochs %epochs% ^
--method %methods% ^
--val_interval %eval% ^
--print_interval 100 ^
--val_print_interval 100 ^
--target_dataset %target_datasets% ^
--target_batch_size %target_batch_size% ^
--crop_val ^
--overlap ^
--weight_samples 1 ^
--filter_unused_labels ^
--uda_lmsq %lmsq% ^
--uda_lIWmsq %ratio% ^
--init_balanced ^
--loss_kd %lkd% ^
--unce ^
--unkd ^
--debug ^
--sample_num 10 ^
--where_to_sim GPU_windows

SET steps=1

SET logdir=logs/
SET out_file=outputs/output_%task%_%names%_step%steps%.txt
echo %out_file%

CALL python -u -m torch.distributed.launch 1> %out_file% 2>&1 ^
--nproc_per_node=1 run_uda.py ^
--batch_size %batch_size% ^
--logdir %logdir% ^
--dataset %source_datasets% ^
--weight_decay %decay% ^
--name %names% ^
--task %task% ^
--step %steps% ^
--lr %lr_step0% ^
--epochs %epochs% ^
--method %methods% ^
--val_interval %eval% ^
--print_interval 100 ^
--val_print_interval 100 ^
--target_dataset %target_datasets% ^
--target_batch_size %target_batch_size% ^
--crop_val ^
--overlap ^
--weight_samples 1 ^
--filter_unused_labels ^
--uda_lmsq %lmsq% ^
--uda_lIWmsq %ratio% ^
--init_balanced ^
--loss_kd %lkd% ^
--unce ^
--unkd ^
--debug ^
--sample_num 10 ^
--where_to_sim GPU_windows

SET steps=2
SET /A epochs=8

SET logdir=logs/
SET out_file=outputs/output_%task%_%names%_step%steps%.txt
echo %out_file%

CALL python -u -m torch.distributed.launch 1> %out_file% 2>&1 ^
--nproc_per_node=1 run_uda.py ^
--batch_size %batch_size% ^
--logdir %logdir% ^
--dataset %source_datasets% ^
--weight_decay %decay% ^
--name %names% ^
--task %task% ^
--step %steps% ^
--lr %lr_step0% ^
--epochs %epochs% ^
--method %methods% ^
--val_interval %eval% ^
--print_interval 100 ^
--val_print_interval 100 ^
--target_dataset %target_datasets% ^
--target_batch_size %target_batch_size% ^
--crop_val ^
--overlap ^
--weight_samples 1 ^
--filter_unused_labels ^
--uda_lmsq %lmsq% ^
--uda_lIWmsq %ratio% ^
--init_balanced ^
--loss_kd %lkd% ^
--unce ^
--unkd ^
--debug ^
--sample_num 10 ^
--where_to_sim GPU_windows