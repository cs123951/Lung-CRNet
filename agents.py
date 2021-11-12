import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

import sys
import random
from public.models import CRNet


class BaseAgent(object):
    def __init__(self, args):
        self.args = args
        self.setup_seed(args['seed'])

        self.model = CRNet

        if self.args['ddp'] and self.args['mode'] == 'train':
            self.model = DDP(self.model)


        if self.args['optim_name'] == "AdamW":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=args['lr'])
        elif self.args['optim_name'] == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args['lr'])

        self.start_epoch, self.best_loss = self.load_checkpoint(args['pretrained_model_path'])
        self.current_epoch = self.start_epoch

        if self.args['mode'] == 'train' and self.args['local_rank'] ==0:
            self.output_directory = self.make_directory()
            self.logger = self.create_logger()
            self.write_config_to_json()
            # self.writer = SummaryWriter(os.path.join(self.output_directory, 'events'))
            # self.writer = open(os.path.join(self.output_directory, 'loss.txt'), "a+")
        if self.args['mode'] == 'test':
            self.output_directory = self.make_directory()

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        start_epoch = 0
        best_loss = 999
        if (self.args['mode'] == 'train' and self.args['resume'] is True) or (self.args['mode'] == 'test'):
            # if resume is true, there must exist file_name
            weights = torch.load(file_name)
            if self.args['mode'] == 'test':
                self.model.load_state_dict(weights['state_dict'])
            elif self.args['ddp'] is True:
                self.model.module.load_state_dict(weights['state_dict'])
            else:
                self.model.load_state_dict(weights['state_dict'])

            start_epoch = weights['epoch']
            best_loss = weights['val_loss']
            print("loading epoch %d " % (start_epoch))

        return start_epoch, best_loss

    def set_train_valid_loader(self):
        raise NotImplementedError

    def set_test_loader(self):
        raise NotImplementedError

    def predict_displacement(self, img_cuda):
        return self.model(img_cuda)


class PairwiseDIR(BaseAgent):
    def __init__(self, args):
        super().__init__(args)


class GroupwiseDIR(BaseAgent):
    def __init__(self, args):
        super().__init__(args)

