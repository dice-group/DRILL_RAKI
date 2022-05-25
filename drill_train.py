"""
====================================================================
Drill -- Deep Reinforcement Learning for Refinement Operators in ALC
====================================================================
Drill with training.
Authors: XXX

This script performs the following computations
1. Parse KG.
2. Generate learning problems.
3. Train DRILL on each learning problems.


=> During training, current state of learning process is displayed periodically.
At the end of the each learning problem, sum of rewards in the first and last three trajectories are shown.
=> Sum of Rewards in first 3 trajectory:[...]
=> Sum of Rewards in last 3 trajectory:[...]
These indicate the learning performance of the agent.


=> As a result the training, a file is created containing all relevant information.
"""
import ontolearn
from core import helper_classes
from core.model import Drill
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.refinement_operators import LengthBasedRefinement
#from ontolearn import KnowledgeBase, LearningProblemGenerator#, DrillAverage, DrillProbabilistic
#from ontolearn.util import sanity_checking_args
import os
import json

import random
import torch
import numpy as np


from argparse import ArgumentParser
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.metrics import F1
from ontolearn.heuristics import Reward
from owlapy.model import OWLOntology, OWLReasoner
from ontolearn.utils import setup_logging

import logging
import operator
import random
import time
from collections import deque
from contextlib import contextmanager
from itertools import islice, chain

from typing import Any, Callable, Dict, FrozenSet, Set, List, Tuple, Iterable, Optional, Generator, SupportsFloat

import torch
from torch import nn
from torch.functional import F
from torch.nn.init import xavier_normal_
from deap import gp, tools, base, creator

from ontolearn.knowledge_base import KnowledgeBase

from ontolearn.core.owl.utils import EvaluatedDescriptionSet, ConceptOperandSorter, OperandSetTransform
from ontolearn.data_struct import PrepareBatchOfTraining, PrepareBatchOfPrediction
from ontolearn.ea_algorithms import AbstractEvolutionaryAlgorithm, EASimple
from ontolearn.ea_initialization import AbstractEAInitialization, EARandomInitialization, EARandomWalkInitialization
from ontolearn.ea_utils import PrimitiveFactory, OperatorVocabulary, ToolboxVocabulary, Tree, escape, ind_to_string, \
    owlliteral_to_primitive_string
from ontolearn.fitness_functions import LinearPressureFitness
from ontolearn.heuristics import OCELHeuristic
from ontolearn.knowledge_base import EvaluatedConcept
from ontolearn.learning_problem import PosNegLPStandard, EncodedPosNegLPStandard
from ontolearn.metrics import Accuracy, F1

from ontolearn.utils import oplogging, create_experiment_folder
from ontolearn.value_splitter import AbstractValueSplitter, BinningValueSplitter, EntropyValueSplitter

from owlapy.render import DLSyntaxObjectRenderer
from owlapy.util import OrderedOWLObject
from sortedcontainers import SortedSet
setup_logging()

logger = logging.getLogger(__name__)

def ClosedWorld_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    from owlapy.owlready2 import OWLOntology_Owlready2
    from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
    from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
    reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=base_reasoner,
                                               negation_default=True)
    return reasoner

random_seed = 1
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


class Trainer:
    def __init__(self, args):
        #sanity_checking_args(args)
        self.args = args

    def save_config(self, path):
        with open(path + '/configuration.json', 'w') as file_descriptor:
            temp = vars(self.args)
            json.dump(temp, file_descriptor)

    def start(self):
        # 1. Parse KG.
        kb = KnowledgeBase(path=self.args.path_knowledge_base, reasoner_factory=ClosedWorld_ReasonerFactory)
        min_num_instances = self.args.min_num_instances_ratio_per_concept * kb.individuals_count()
        max_num_instances = self.args.max_num_instances_ratio_per_concept * kb.individuals_count()

        # 2. Generate Learning Problems.
        lp = LearningProblemGenerator(knowledge_base=kb,
                                      min_length=self.args.min_length,
                                      max_length=self.args.max_length,
                                      min_num_instances=min_num_instances,
                                      max_num_instances=max_num_instances)
        balanced_examples = lp.get_balanced_n_samples_per_examples(
            n=self.args.num_of_randomly_created_problems_per_concept,
            min_length=self.args.min_length,
            max_length=self.args.max_length,
            min_num_problems=self.args.min_num_concepts,
            num_diff_runs=self.args.min_num_concepts // 2)
        drill = Drill(knowledge_base=kb, path_of_embeddings=self.args.path_knowledge_base_embeddings,
                      refinement_operator=LengthBasedRefinement(knowledge_base=kb), quality_func=F1(),
                      reward_func=Reward(),
                      batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                      pretrained_model_path=self.args.pretrained_drill_avg_path, verbose=self.args.verbose,
                      max_len_replay_memory=self.args.max_len_replay_memory, epsilon_decay=self.args.epsilon_decay,
                      num_epochs_per_replay=self.args.num_epochs_per_replay,
                      num_episodes_per_replay=self.args.num_episodes_per_replay, learning_rate=self.args.learning_rate,
                      num_of_sequential_actions=self.args.num_of_sequential_actions, num_episode=self.args.num_episode)

        drill.train(balanced_examples)
        drill.save_weights()
        for result_dict, learning_problem in zip(
                drill.fit_from_iterable(balanced_examples, max_runtime=self.args.max_test_time_per_concept),
                balanced_examples):
            target_class_expression, sampled_positive_examples, sampled_negative_examples = learning_problem
            print(f'\nTarget Class Expression:{target_class_expression}')
            print(f'| sampled E^+|:{len(sampled_positive_examples)}\t| sampled E^-|:{len(sampled_negative_examples)}')
            for k, v in result_dict.items():
                print(f'{k}:{v}')

if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", default='KGs/Family/family-benchmark_rich_background.owl')
    parser.add_argument("--path_knowledge_base_embeddings",
                        default='embeddings/ConEx_Family/ConEx_entity_embeddings.csv')
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of cpus used during batching')

    # Concept Generation Related
    parser.add_argument("--min_num_concepts", type=int, default=1)
    parser.add_argument("--min_length", type=int, default=4, help='Min length of concepts to be used')
    parser.add_argument("--max_length", type=int, default=5, help='Max length of concepts to be used')
    parser.add_argument("--min_num_instances_ratio_per_concept", type=float, default=.01)  # %1
    parser.add_argument("--max_num_instances_ratio_per_concept", type=float, default=.60)  # %30
    parser.add_argument("--num_of_randomly_created_problems_per_concept", type=int, default=1)

    # DQL related
    parser.add_argument("--gamma", type=float, default=.99, help='The discounting rate')
    parser.add_argument("--num_episode", type=int, default=5, help='Number of trajectories created for a given lp.')
    parser.add_argument("--epsilon_decay", type=float, default=.01, help='Epsilon greedy trade off per epoch')
    parser.add_argument("--max_len_replay_memory", type=int, default=1024,
                        help='Maximum size of the experience replay')
    parser.add_argument("--num_epochs_per_replay", type=int, default=3,
                        help='Number of epochs on experience replay memory')
    parser.add_argument("--num_episodes_per_replay", type=int, default=10, help='Number of episodes per repay')
    parser.add_argument('--num_of_sequential_actions', type=int, default=3, help='Length of the trajectory.')
    parser.add_argument('--relearn_ratio', type=int, default=1, help='# of times lps are reused.')
    parser.add_argument('--use_illustrations', default=True, type=eval, choices=[True, False])
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
