"""
Deploy our approach
# positive examples as a string
http://www.biopax.org/examples/glycolysis#complex273,http://www.biopax.org/examples/glycolysis#complex282

http://www.biopax.org/examples/glycolysis#complex241

"""

import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify

from typing import Dict

import pandas as pd
import torch
import gradio as gr
import tempfile
from argparse import ArgumentParser

import ontolearn
from core.model import Drill
from ontolearn.refinement_operators import LengthBasedRefinement
from ontolearn.metrics import F1
from ontolearn.knowledge_base import KnowledgeBase

from owlapy.render import DLSyntaxObjectRenderer
from owlapy.model import OWLNamedIndividual, OWLOntology, OWLReasoner, IRI
from owlapy.owlready2 import OWLOntology_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
app = Flask(__name__)
kb = None
drill = None
renderer = DLSyntaxObjectRenderer()


def load_model(kb, args) -> Drill:
    return Drill(knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
                 refinement_operator=LengthBasedRefinement(knowledge_base=kb), quality_func=F1(),
                 pretrained_model_path=args.pretrained_drill_avg_path, verbose=args.verbose)


def is_input_valid(pos: str, neg: str):
    return len(pos) > 0 and len(pos) > 0


def ClosedWorld_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    assert isinstance(onto, OWLOntology_Owlready2)
    return OWLReasoner_FastInstanceChecker(ontology=onto,
                                           base_reasoner=OWLReasoner_Owlready2_TempClasses(ontology=onto),
                                           negation_default=True)


@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    print(json_)

    # Map str IRI into OWLNamedIndividual
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, json_["positives"])))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, json_["negatives"])))
    drill.fit(typed_pos, typed_neg, max_runtime=3)

    if json_.get("save_predictions", None):
        tmp = tempfile.NamedTemporaryFile()
        drill.save_best_hypothesis(10, tmp.name)
    data = []
    for i in drill.best_hypotheses(10):
        str_rep, f1_score = renderer.render(i.concept), i.quality
        data.append([str_rep, f1_score])

    # Converting to int from int64
    return jsonify({"prediction": data})


def run(args):
    # Load data
    global kb
    kb = KnowledgeBase(path=args.path_knowledge_base, reasoner_factory=ClosedWorld_ReasonerFactory)

    # Load model
    global drill
    drill = load_model(kb, args)
    # Lunch
    app.run(host=args.server_name, port=args.server_port, debug=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_knowledge_base", type=str, default='KGs/Biopax/biopax.owl')
    parser.add_argument("--path_knowledge_base_embeddings", type=str,
                        default='embeddings/ConEx_Biopax/ConEx_entity_embeddings.csv')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of cpus used during batching')
    parser.add_argument("--verbose", type=int, default=0, help='Higher integer reflects more info during computation')

    # Concept Generation Related
    parser.add_argument('--num_of_sequential_actions', type=int, default=3, help='Length of the trajectory.')

    parser.add_argument('--pretrained_drill_avg_path', type=str,
                        default='pre_trained_agents/Biopax/DrillHeuristic_averaging/DrillHeuristic_averaging.pth',
                        help='Provide a path of .pth file')
    # Concept Learning Testing
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    parser.add_argument('--max_test_time_per_concept', type=int, default=3, help='Max. runtime during testing')
    # Inference Related
    parser.add_argument("--topk", type=int, default=10,
                        help='Return top k concepts')
    parser.add_argument('--use_search', default='None', help='None,SmartInit')
    parser.add_argument('--server_port', default=7860, type=int)
    parser.add_argument('--server_name', default="0.0.0.0")
    run(parser.parse_args())
