# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
import argparse        
import yaml
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lasr.utils.generater import BaseConfig
from lasr.utils.data_utils import get_s2s_inout

class LightModelFace(pl.LightningModule):
    def __init__(
            self, 
            model, 
            criterion, 
            optim,
            scheduler,
            tokenizer,
            model_config=None, 
            criterion_config=None, 
            optim_config=None,
            tokenizer_config=None
        ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optim = optim
        self.scheduler = scheduler
        self.tokenizer = tokenizer

        self.model_config = model_config
        self.criterion_config = criterion_config
        self.optim_config = optim_config
        self.tokenizer_config = tokenizer_config

        self.save_hyperparameters("model_config", "criterion_config", "optim_config", "tokenizer_config")

    def training_step(self, batch, batch_idx):
        data = self.pack_data(batch)
        data.update(self.model.train_forward(data)) 
        data.update(self.criterion.train_forward(data)) 
        loss = data["loss_main"]
        self.log("LR", self.optim.param_groups[0]['lr'])
        for k, v in data.items():
            if isinstance(v, (int, float)) or v.numel() == 1:
                 self.log(k, v)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data = self.pack_data(batch)
        data.update(self.model.valid_forward(data)) 
        data.update(self.criterion.valid_forward(data)) 
        # loss = data["loss_main"]
        self.log("global_step", self.global_step, batch_size=1, sync_dist=True)
        for k, v in data.items():
            if isinstance(v, (int, float)) or v.numel() == 1:
                 self.log("valid_" + k, v, batch_size=1, sync_dist=True)

    def configure_optimizers(self):      
        if self.scheduler is None:  
            return self.optim
        else:
            return {
                "optimizer": self.optim,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "step",
                }
            }
        
    def pack_data(self, data):
        x = data["wav_array"]
        xlen = data["wav_len"]
        y = data["token_id"]
        ylen = data["token_len"]
        # 获取attention分支的输入输出
        ys_in, ys_out = get_s2s_inout(
            y, 
            sos=self.tokenizer.ID_VALUE_SOS,
            eos=self.tokenizer.ID_VALUE_EOS,
            pad=self.tokenizer.ID_VALUE_PAD,
            ignore=self.tokenizer.ID_VALUE_IGNORE            
        )
        # 获取CTC分支的标签，即把PAD转换为IGNORE
        y[y==self.tokenizer.ID_VALUE_PAD] = self.tokenizer.ID_VALUE_IGNORE
        return {
            "x": x, 
            "xlen": xlen, 
            "ys_in": ys_in,
            "ylen": ylen,
            "att_label": ys_out,
            "ctc_label": y,
        }
        
    def get_callbacks(self):
        
        checkpoint_callback_top = ModelCheckpoint(
            save_top_k=10,
            monitor="valid_loss_main",
            mode="min",
            # dirpath="my/path/",
            filename="best-val-{valid_loss_main:.6f}-{epoch:02d}",
            )

        # saves last-K checkpoints based on "global_step" metric
        # make sure you log it inside your LightningModule
        checkpoint_callback_last = ModelCheckpoint(
            save_top_k=10,
            monitor="global_step",
            mode="max",
            # dirpath="my/path/",
            filename="last-step-{epoch:02d}-{global_step}",
        )

        return [checkpoint_callback_top, checkpoint_callback_last]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_dir", default='exp', type=str)
    parser.add_argument("-config")
    parser.add_argument("-num_gpu", default=-1, type=int,
                    metavar='N', help='number of gpu to use, -1 means cpu, 0 means no paraller')
    parser.add_argument('-num_epochs', default=50, type=int,
                    metavar='N', help='train_epochs')
    parser.add_argument('-resume_ckpt', default=None, type=str,
                    help='resume the training ckpt')    

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f) 

    train_data_config = config['train_data_config']
    valid_data_config = config['valid_data_config'] 
    model_config = config['model_config'] 
    opt_config = config["opti_config"]
    criterion_config = config["criterion_config"]
    tokenizer_config = config["tokenizer_config"]

    tokenizer = BaseConfig(**tokenizer_config).generateExample()
    train_dataset = BaseConfig(**train_data_config).generateExample(tokenizer=tokenizer)
    valid_dataset = BaseConfig(**valid_data_config).generateExample(tokenizer=tokenizer)

    output_dim = tokenizer.dict_size()
    model_config["kwargs"]["odim"] = output_dim
    criterion_config["kwargs"]["size"] = output_dim
    criterion_config["kwargs"]["padding_idx"] = tokenizer.ID_VALUE_IGNORE


    model = BaseConfig(**model_config).generateExample()
    criterion = BaseConfig(**criterion_config).generateExample()
    optim = BaseConfig(name=opt_config['name'], kwargs=opt_config['kwargs']).generateExample(model.parameters())
    if 'scheduler' in opt_config:
        scheduler = BaseConfig(**opt_config['scheduler']).generateExample(optim)
    else:
        scheduler = None


    
    lmodel = LightModelFace(
        model, 
        criterion, 
        optim,
        scheduler, 
        tokenizer,
        model_config, 
        criterion_config, 
        opt_config,
        tokenizer_config
    ) 

    trainer = pl.Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.num_epochs, min_epochs=1, #同理还有step和time
        accumulate_grad_batches=1, gradient_clip_val=5, #梯度裁剪和梯度平均
        benchmark=None, # True for fixed input size
        callbacks=lmodel.get_callbacks(), #靠回调函数来进行各种增强功能
        check_val_every_n_epoch=1, #CheckPoint参数
        # val_check_interval=1, #大于1则对应batch数，小于1则对应训练epoch的进度
        enable_checkpointing=True, # 默认只存储唯一一个checkpoints，定义ModelCheckpoint callback来修改存储方式
        fast_dev_run=False, #Debug使用，只跑一个bacth
        limit_train_batches=None, #Debug使用，小于1是比例，大于1是batch数量，同理还有limit_test/val_batches
        num_nodes=1, #分布式参数
        strategy="ddp", # ddp
        accelerator="gpu", # gpu 
        devices=args.num_gpu, # 5
        sync_batchnorm=False,
        use_distributed_sampler=True, #是否自动添加分布式训练采样，False则需要自己写分布式采样的方式
        precision=32, #训练精度
        enable_progress_bar=True, #进度条
        logger=True, #自定义Logger，True为TensorBoard，可以修改为其他的自定义的
        inference_mode=False, #相当于no_grad
        )
    
    train_dataset.load_check_data()
    valid_dataset.load_check_data()

    print(model)
    print(criterion)
    print(optim)
    print(scheduler)
    print(tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=1, # Dataset has maken batch according to the audio duration
        shuffle=True, 
        num_workers=16, 
        collate_fn=train_dataset.collate_fn
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=16, 
        collate_fn=valid_dataset.collate_fn
    )
 
    trainer.fit(
        model=lmodel, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=valid_dataloader,
        ckpt_path=args.resume_ckpt
    )

    # trainer.test(model=lmodel, dataloaders=valid_dataloader)


if __name__ == "__main__":
    main()