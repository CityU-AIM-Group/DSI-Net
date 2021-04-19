class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, avg_mom=0.5):
        self.avg_mom = avg_mom
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0 # running average of whole epoch
        self.smooth_avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.smooth_avg = val if self.count == 0 else self.avg*self.avg_mom + val*(1-self.avg_mom)
        self.avg = self.sum / self.count