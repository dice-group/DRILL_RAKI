"""
====================================================================
Drill -- Deep Reinforcement Learning for Refinement Operators in ALC
====================================================================
Reproducing our experiments Experiments

This script performs the following computations
1. Parse KG.
2. Load learning problems LP= {(E^+,E^-)...]

3. Initialize models .
    3.1. Initialize DL-learnerBinder objects to communicate with DL-learner binaries.
    3.2. Initialize DRILL.
4. Provide models + LP to Experiments object.
    4.1. Each learning problem provided into models
    4.2. Best hypothesis/predictions of models given E^+ and E^- are obtained.
    4.3. F1-score, Accuracy, Runtimes and Number description tested information stored and serialized.
"""
import ontolearn
from core import helper_classes
from core.model import Drill
from core.static_funcs import ClosedWorld_ReasonerFactory
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.refinement_operators import LengthBasedRefinement, ModifiedCELOERefinement
from ontolearn.metrics import Accuracy, F1

import random
from ontolearn.experiments import Experiments
from ontolearn.binders import DLLearnerBinder
import pandas as pd
from argparse import ArgumentParser
import os
import json
import time
from typing import Dict, List, AnyStr

full_computation_time = time.time()


def sanity_checking_args(args):
    try:
        assert os.path.isfile(args.path_knowledge_base)
    except AssertionError as e:
        print(f'--path_knowledge_base ***{args.path_knowledge_base}*** does not lead to a file.')
        exit(1)
    assert os.path.isfile(args.path_knowledge_base_embeddings)
    assert os.path.isfile(args.path_knowledge_base)


def learning_problem_parser_from_json(path) -> List:
    """ Load Learning Problems from Json into List"""
    # (1) Read json file into a python dictionary
    with open(path) as json_file:
        storage = json.load(json_file)
    # (2) json file contains stores each learning problem as a value in a key called "problems"
    assert len(storage) == 1
    # (3) Validate that we have at least single learning problem
    assert len(storage['problems']) > 0
    problems = storage['problems']
    # (4) Parse learning problems with sanity checking
    problems: Dict[AnyStr, Dict[AnyStr, List]]  # , e.g.
    """ {'Aunt'}: {'positive_examples':[...], 'negative_examples':[...], 'ignore_concepts':[...] """
    class_expression_learning_problems = []
    for target_name, lp in problems.items():
        assert 'positive_examples' in lp
        assert 'negative_examples' in lp

        positive_examples = set(lp['positive_examples'])
        negative_examples = set(lp['negative_examples'])

        if 'ignore_concepts' in lp:
            ignore_concepts = set(lp['ignore_concepts'])
        else:
            ignore_concepts = set()
        class_expression_learning_problems.append({
            'target_concept': target_name,
            'positive_examples': positive_examples,
            'negative_examples': negative_examples,
            'ignore_concepts': ignore_concepts
        })

    return class_expression_learning_problems


def remedy_lps_wo_negatives(kb, problems):
    for ith, lp_dict in enumerate(problems):
        if len(lp_dict['negative_examples']) == 0:
            print(f'{ith} learning problem without negatives is remedied.')
            lp_dict['negative_examples'] = kb.individuals.difference(lp_dict['positive_examples'])
    return problems


def start(args):
    kb = KnowledgeBase(path=args.path_knowledge_base, reasoner_factory=ClosedWorld_ReasonerFactory)

    problems = learning_problem_parser_from_json(args.path_lp)
    # Remedy empty negatives
    problems = remedy_lps_wo_negatives(kb, problems)

    print(f'Number of problems {len(problems)} on {kb}')

    models = []
    # Initialize models
    if args.path_dl_learner:
        models.append(
            DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='celoe'))
        models.append(DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='ocel'))
        models.append(DLLearnerBinder(binary_path=args.path_dl_learner, kb_path=args.path_knowledge_base, model='eltl'))
    drill = Drill(knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
                  refinement_operator=ModifiedCELOERefinement(knowledge_base=kb),
                  quality_func=F1(), batch_size=args.batch_size, num_workers=args.num_workers,
                  pretrained_model_path=args.pretrained_drill_avg_path, verbose=args.verbose,
                  num_of_sequential_actions=args.num_of_sequential_actions)

    drill.name += '_ModifiedCELOERefinement'
    models.append(drill)
    models.append(Drill(knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
                        refinement_operator=LengthBasedRefinement(knowledge_base=kb),
                        quality_func=F1(), batch_size=args.batch_size, num_workers=args.num_workers,
                        pretrained_model_path=args.pretrained_drill_avg_path, verbose=args.verbose,
                        num_of_sequential_actions=args.num_of_sequential_actions))

    time_kg_processing = time.time() - full_computation_time
    print(f'KG preprocessing took : {time_kg_processing}')
    drill.time_kg_processing = time_kg_processing

    problems = [(i['target_concept'], i['positive_examples'], i['negative_examples']) for i in problems]
    Experiments(max_test_time_per_concept=args.max_test_time_per_concept).start(dataset=problems,models=models)


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str, default='KGs/Biopax/biopax.owl')
    parser.add_argument("--path_lp", type=str, default='LPs/Biopax/lp.json')
    parser.add_argument("--path_knowledge_base_embeddings", type=str,
                        default='embeddings/ConEx_Biopax/ConEx_entity_embeddings.csv')
    # The next two params shows the flexibility of our framework as agents can be continuously trained
    parser.add_argument('--pretrained_drill_avg_path', type=str,
                        default='pre_trained_agents/Biopax/DrillHeuristic_averaging/DrillHeuristic_averaging.pth',
                        help='Provide a path of .pth file')
    parser.add_argument("--refinement_operator", type=str, default='LengthBasedRefinement',
                        choices=['ModifiedCELOERefinement', 'LengthBasedRefinement'])

    parser.add_argument('--num_workers', type=int, default=1, help='Number of cpus used during batching')
    parser.add_argument("--verbose", type=int, default=0, help='Higher integer reflects more info during computation')

    # Concept Generation Related
    parser.add_argument("--min_num_concepts", type=int, default=1)
    parser.add_argument("--min_length", type=int, default=3, help='Min length of concepts to be used')
    parser.add_argument("--max_length", type=int, default=5, help='Max length of concepts to be used')
    parser.add_argument("--min_num_instances_ratio_per_concept", type=float, default=.01)  # %1
    parser.add_argument("--max_num_instances_ratio_per_concept", type=float, default=.90)  # %30
    parser.add_argument("--num_of_randomly_created_problems_per_concept", type=int, default=1)
    # DQL related
    parser.add_argument("--num_episode", type=int, default=1, help='Number of trajectories created for a given lp.')
    parser.add_argument('--relearn_ratio', type=int, default=1,
                        help='Number of times the set of learning problems are reused during training.')
    parser.add_argument("--gamma", type=float, default=.99, help='The discounting rate')
    parser.add_argument("--epsilon_decay", type=float, default=.01, help='Epsilon greedy trade off per epoch')
    parser.add_argument("--max_len_replay_memory", type=int, default=1024,
                        help='Maximum size of the experience replay')
    parser.add_argument("--num_epochs_per_replay", type=int, default=2,
                        help='Number of epochs on experience replay memory')
    parser.add_argument("--num_episodes_per_replay", type=int, default=10, help='Number of episodes per repay')
    parser.add_argument('--num_of_sequential_actions', type=int, default=3, help='Length of the trajectory.')

    # NN related
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=int, default=.01)

    # Concept Learning Testing
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    parser.add_argument('--max_test_time_per_concept', type=int, default=3, help='Max. runtime during testing')
    # Binaries for DL-learner
    parser.add_argument("--path_dl_learner", type=str, default=None, help='Path of dl-learner binaries.')

    start(parser.parse_args())
