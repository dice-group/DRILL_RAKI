"""
Deploy our approach
# positive examples as a string
http://www.biopax.org/examples/glycolysis#complex273,http://www.biopax.org/examples/glycolysis#complex282

http://www.biopax.org/examples/glycolysis#complex241

"""
from typing import Dict

import pandas as pd
import torch
import json
import gradio as gr

import os
import json
import io
import threading
from pathlib import Path
import tempfile
import time
from datetime import datetime
from argparse import ArgumentParser
from functools import wraps, update_wrapper
from flask import Flask, request, Response, abort
from flask import make_response

import ontolearn
from core.model import Drill
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.metrics import F1
from ontolearn.utils import setup_logging
from ontolearn.knowledge_base import KnowledgeBase

from owlapy.render import DLSyntaxObjectRenderer
from owlapy.model import OWLOntology, OWLReasoner
from owlapy.model import OWLNamedIndividual
from owlapy.model import IRI
from owlapy.owlready2 import OWLOntology_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker


def load_target_class_expressions_and_instance_idx_mapping(args):
    """

    :param args:
    :return:
    """
    # target_class_expressions Must be empty and must be filled in an exactorder
    target_class_expressions = []
    df = ddf.read_csv(args['path_of_experiment_folder'] + '/target_class_expressions.csv', dtype={'label_id': 'int',
                                                                                                  'name': 'object',
                                                                                                  'str_individuals': 'object',
                                                                                                  'idx_individuals': 'object',
                                                                                                  'atomic_expression': 'object',
                                                                                                  'concepts': 'object',
                                                                                                  'filler': 'object',
                                                                                                  'role': 'object',
                                                                                                  })
    df = df.compute(scheduler='processes').set_index('Unnamed: 0')

    print(df.head())

    with open(args['path_of_experiments'] + '/target_class_expressions.json', 'r') as f:
        for k, v in json.load(f).items():
            k: str  # k denotes k.th label of target expression, json loading type conversion from int to str appreantly
            v: dict  # v contains info for Target Class Expression Object
            assert isinstance(k, str)
            assert isinstance(v, dict)
            try:
                k = int(k)
            except ValueError:
                print(k)
                print('Tried to convert to int')
                exit(1)
            try:

                assert k == v['label_id']
            except AssertionError:
                print(k)
                print(v['label_id'])
                exit(1)

            t = TargetClassExpression(label_id=v['label_id'],
                                      name=v['name'],
                                      idx_individuals=frozenset(v['idx_individuals']),
                                      expression_chain=v['expression_chain'])
            assert len(t.idx_individuals) == len(v['idx_individuals'])

            target_class_expressions.append(t)

    instance_idx_mapping = dict()
    with open(args['path_of_experiments'] + '/instance_idx_mapping.json', 'r') as f:
        instance_idx_mapping.update(json.load(f))

    return target_class_expressions, instance_idx_mapping


def load_model(kb, args: Dict) -> torch.nn.Module:
    return Drill(knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
                 refinement_operator=LengthBasedRefinement(knowledge_base=kb), quality_func=F1(),
                 batch_size=args.batch_size, num_workers=args.num_workers,
                 pretrained_model_path=args.pretrained_drill_avg_path, verbose=args.verbose,
                 num_of_sequential_actions=args.num_of_sequential_actions)


def is_input_valid(pos: str, neg: str):
    return len(pos) > 0 and len(pos) > 0


def launch_service(drill, kb, max_test_time_per_concept, server_name: str = None, server_port: int = None):
    renderer=DLSyntaxObjectRenderer()

    def predict(positive_examples, negative_examples, save_predictions):
        if is_input_valid(positive_examples, negative_examples):
            pos_str = positive_examples.split(",")
            neg_str = negative_examples.split(",")

            if len(pos_str) < 20:
                s = f'E^+:{",".join(pos_str)}\nE^-:{",".join(neg_str)}\n'
            else:
                s = f'|E^+|:{len(pos_str)}\n|E^-|:{len(neg_str)}\n'
            # Map str IRI into OWLNamedIndividual
            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, pos_str)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, neg_str)))
            drill.fit(typed_pos, typed_neg, max_runtime=max_test_time_per_concept)

            if save_predictions:
                tmp = tempfile.NamedTemporaryFile()
                drill.save_best_hypothesis(10, tmp.name)

            data=[]
            for i in drill.best_hypotheses(10):
                str_rep, f1_score=renderer.render(i.concept), i.quality
                data.append([str_rep, f1_score])
            return s, pd.DataFrame(data=data,columns=["Concept", "F1-score"])
        else:
            return f"Invalid Input:{positive_examples}***{negative_examples}", pd.DataFrame([0, 0, 0])

    gr.Interface(
        fn=predict,
        inputs=[gr.inputs.Textbox(lines=5, placeholder=None, label='Positive Examples'),
                gr.inputs.Textbox(lines=5, placeholder=None, label='Negative Examples'),
                "checkbox"],
        outputs=[gr.outputs.Textbox(label='Learning Problem'),
                 gr.outputs.Dataframe(label='Predictions', type='pandas')],
        title='Class Expression Learning',
        description='Click Random Examples & Submit.').launch(server_name=server_name, server_port=server_port,
                                                              share=False)


def ClosedWorld_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
    reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=base_reasoner,
                                               negation_default=True)
    return reasoner


def run(args):
    # Load data
    kb = KnowledgeBase(path=args.path_knowledge_base, reasoner_factory=ClosedWorld_ReasonerFactory)
    drill = load_model(kb, args)
    # Individuals
    launch_service(drill, kb, max_test_time_per_concept=args.max_test_time_per_concept, server_name=args.server_name,
                   server_port=args.server_port)


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str, default='KGs/Biopax/biopax.owl')
    parser.add_argument("--path_knowledge_base_embeddings", type=str,
                        default='embeddings/ConEx_Biopax/ConEx_entity_embeddings.csv')
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

    # The next two params shows the flexibility of our framework as agents can be continuously trained
    parser.add_argument('--pretrained_drill_avg_path', type=str,
                        default='pre_trained_agents/Biopax/DrillHeuristic_averaging/DrillHeuristic_averaging.pth',
                        help='Provide a path of .pth file')
    # NN related
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=int, default=.01)
    parser.add_argument("--drill_first_out_channels", type=int, default=32)

    # Concept Learning Testing
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    parser.add_argument('--max_test_time_per_concept', type=int, default=3, help='Max. runtime during testing')

    # Inference Related
    parser.add_argument("--topK", type=int, default=100,
                        help='Test the highest topK target expressions')
    parser.add_argument("--use_multiprocessing_at_parsing", type=int,
                        default=0, help='1 or 0')
    parser.add_argument('--use_search', default='None', help='None,SmartInit')
    parser.add_argument('--server_port', default=7860, type=int)
    parser.add_argument('--server_name', default="0.0.0.0")
    run(parser.parse_args())
