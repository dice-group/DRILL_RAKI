"""
====================================================================
Drill -- Deep Reinforcement Learning for Refinement Operators in ALC
====================================================================
Drill with training.
Authors: Caglar Demir

This script performs the following computations via the Trainer Class
1. Parse KG.
2. Generate learning problems.
3. Train DRILL on each learning problems.
"""
import logging
from argparse import ArgumentParser
from ontolearn.utils import setup_logging
from core.trainer import Trainer
setup_logging()
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", default='/home/demir/Desktop/Siemens/KG/cybersecurity_bundle/short_cyber_new.owl')
    parser.add_argument("--path_knowledge_base_embeddings",
                        default='/home/demir/Desktop/Siemens/ConEx_entity_embeddings.csv')
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of cpus used during batching')
    # Concept Generation Related
    parser.add_argument("--min_num_concepts", type=int, default=1)
    parser.add_argument("--min_length", type=int, default=1, help='Min length of concepts to be used')
    parser.add_argument("--max_length", type=int, default=1, help='Max length of concepts to be used')
    parser.add_argument("--min_num_instances_ratio_per_concept", type=float, default=None)  # %1
    parser.add_argument("--max_num_instances_ratio_per_concept", type=float, default=None)  # %30
    parser.add_argument("--num_of_randomly_created_problems_per_concept", type=int, default=0)
    parser.add_argument("--refinement_operator", type=str, default='LengthBasedRefinement',
                        choices=['ModifiedCELOERefinement', 'LengthBasedRefinement'])

    # DQL related
    parser.add_argument("--gamma", type=float, default=.99, help='The discounting rate')
    parser.add_argument("--num_episode", type=int, default=5, help='Number of trajectories created for a given lp.')
    parser.add_argument("--epsilon_decay", type=float, default=.01, help='Epsilon greedy trade off per epoch')
    parser.add_argument("--max_len_replay_memory", type=int, default=1024,
                        help='Maximum size of the experience replay')
    parser.add_argument("--num_epochs_per_replay", type=int, default=3,
                        help='Number of epochs on experience replay memory')
    parser.add_argument("--num_episodes_per_replay", type=int, default=10, help='Number of episodes per repay')
    parser.add_argument('--num_of_sequential_actions', type=int, default=1, help='Length of the trajectory.')
    parser.add_argument('--relearn_ratio', type=int, default=1, help='# of times lps are reused.')
    parser.add_argument('--use_illustrations', default=False, type=eval, choices=[True, False])
    parser.add_argument('--use_target_net', default=False, type=eval, choices=[True, False])

    # The next two params shows the flexibility of our framework as agents can be continuously trained
    parser.add_argument('--pretrained_drill_avg_path', type=str,
                        default='', help='Provide a path of .pth file')
    # NN related
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=int, default=.01)
    parser.add_argument("--drill_first_out_channels", type=int, default=32)

    # Concept Learning Testing
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    parser.add_argument('--max_test_time_per_concept', type=int, default=3, help='Max. runtime during testing')

    trainer = Trainer(parser.parse_args())
    trainer.start()
