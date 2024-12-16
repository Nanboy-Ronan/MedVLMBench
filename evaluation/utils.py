import os
import math
import time
from collections import defaultdict, deque
import datetime
import numpy as np

import torch
import torch.distributed as dist
from torch import inf
from evaluation.vqa import VQABenchmark


def get_benchmark(args, dataset=None):
    benchmark = VQABenchmark(args=args, dataset=dataset, logger=args.logger)
    return benchmark