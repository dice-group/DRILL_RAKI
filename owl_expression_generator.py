import logging
from ontolearn.utils import setup_logging
from core.helper_classes import ExpressionGenerator
from argparse import ArgumentParser
setup_logging()
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_knowledge_base", default='KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--path_to_store", default='generated_concepts')
    parser.add_argument("--num_problems", type=int, default=1)
    parser.add_argument("--num_diff_runs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=6)
    parser.add_argument("--min_length", type=int, default=3)
    parser.add_argument("--min_num_instances", type=int, default=4)
    parser.add_argument("--max_num_instances", type=int, default=200)
    ExpressionGenerator(args=parser.parse_args())
