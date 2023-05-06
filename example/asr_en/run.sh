PLAT_ROOT=../../
export PYTHONPATH=$PLAT_ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

tag=ls_swbd_cmv_ami_chime

mkdir -p exp/train_${tag}
cp conf/config_${tag}.yaml exp/train_${tag}

python $PLAT_ROOT/bin/train_lighting.py \
    -config conf/config_${tag}.yaml \
    -exp_dir exp/train_${tag} \
    -num_epochs 100 \
    -num_gpu 6 \
    -acc_grads 4 \


avg=5

python $PLAT_ROOT/bin/decode_lighting.py \
    -train_config exp/train_${tag}/lightning_logs/version_0/hparams.yaml \
    -decode_config conf/decode.yaml \
    -model_path exp/train_${tag}/lightning_logs/version_0/checkpoints/ \
    -avg $avg \
    -device "cuda:0" \
    -choose "last" \
    -output_file exp/train_${tag}/decode_avg${avg}.txt  > exp/train_${tag}/decode_avg${avg}.log

