from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, 
                 model_size, factor, warm_step, offset=0, offstep=0):
        self.warmup = warm_step
        self.factor = factor
        self.model_size = model_size
        self.offset = offset
        self.offstep= offstep
        super(WarmupScheduler, self).__init__(optimizer, -1, False)


    def get_lr(self):
        return [
            self.offset + self.factor
            * self.model_size ** (-0.5)
            * min(self._step_count  ** (-0.5), self._step_count  * self.warmup ** (-1.5)) 
            for group in self.optimizer.param_groups
        ]

