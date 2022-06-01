import argparse
import sys
from copy import copy
import importlib
from core import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-C", "--config", help="config filename", default='core.configs.fam')
    parser.add_argument("-s", "--seed", type=int, help="a seed for the random number generator", default=0)
    parser_args, _ = parser.parse_known_args(sys.argv)
    # See  parser_args.config for default params
    cfg = copy(importlib.import_module(parser_args.config).cfg)
    trainer = Trainer(cfg)
    trainer.start()
