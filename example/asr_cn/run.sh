PLAT_ROOT=../../
export PYTHONPATH=$PLAT_ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

tag=baseline

mkdir -p exp/train_${tag}
cp conf/config_${tag}.yaml exp/train_${tag}

python $PLAT_ROOT/bin/train_lighting.py \
    -config conf/config_${tag}.yaml \
    -exp_dir exp/train_${tag} \
    -num_epochs 100 \
    -num_gpu 2 \
    # -resume_ckpt exp/train_aishell1_conformer_spec_batch96/lightning_logs/version_0/checkpoints/last-step-epoch=79-global_step=21680.0.ckpt

python $PLAT_ROOT/bin/decode_lighting.py \
    -train_config exp/train_${tag}/lightning_logs/version_0/hparams.yaml \
    -decode_config conf/decode.yaml \
    -model_path exp/train_${tag}/lightning_logs/version_0/checkpoints/ \
    -output_file exp/train_${tag}/decode.txt  > exp/train_${tag}/decode.log