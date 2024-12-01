# -*- coding: utf-8 -*-
"""
@author: PC
Update Time: 2024-12-01
"""
from depend.baselogic import BaseLogic
from depend.lcii import LatentContextIIGRU
from depend.argumentparser import AP

class Entry:
    def __init__(self):
        self.dataset = None
        self.switch_plot = None
        self.fusion_way = None
        self.strategy = None
        self.window = None
        self.lo_score = None
        self.sh_score = None
        self.embed_size = None
        self.batch_size = None
        self.learning_rate = None
        self.dropout = None
        self.max_epoch = None
        self.threshold = None
        self.use_fc = None
        self.switch_initial_state = None

    def main(self):
        ap = AP(self)
        ap.config_once()
        BaseLogic.check_gpu_running()
        lcii = LatentContextIIGRU(self)
        lcii.main()

if __name__ == '__main__':
    entry = Entry()
    entry.main()