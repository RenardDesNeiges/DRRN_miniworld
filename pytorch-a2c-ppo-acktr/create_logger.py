import os
import logging
import time
from os.path import dirname, realpath, join, expanduser, normpath,isdir,split
def create_logger(root_out_path):
    #set up logger
    if not os.path.exists(root_out_path):
        os.makedirs(root_out_path)
    assert os.path.exists(root_out_path), '{} does not exits'.format(root_out_path)
    log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(root_out_path,log_file),format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger



def make_path(output_path):
    if not isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    return output_path


class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

        def __getattr__(self, attr):
            return self.get(attr)

        def __setattr__(self, key, value):
            self.__setitem__(key, value)

        def __setitem__(self, key, value):
            super(Map, self).__setitem__(key, value)
            self.__dict__.update({key: value})

        def __delattr__(self, item):
            self.__delitem__(item)

        def __delitem__(self, key):
            super(Map, self).__delitem__(key)
            del self.__dict__[key]




class Logger_tensorboard:
    def __init__(self, out_dir, out_name='log_tb', use_tensorboard=False):
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.out_dir = out_dir
            self.writer = SummaryWriter(out_dir + out_name)
            self.add_losses = self.add_scalars
        else:
            # # make a dummy functions if we do not want to write to this logger.
            self.add_losses = lambda *args: 1

    def close(self):
        if self.use_tensorboard:
            self.writer.close()

    def add_scalars(self, data, n_iter):
        # # assume that we want to write in the logger (no check!).
        for key, value in data.items():
            self.writer.add_scalar(key, value, n_iter)


