"""
Deploy our approach
# positive examples as a string
http://www.biopax.org/examples/glycolysis#complex273,http://www.biopax.org/examples/glycolysis#complex282

http://www.biopax.org/examples/glycolysis#complex241

"""
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

def load_model(kb, args) -> Drill:
    return Drill(knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
                 refinement_operator=LengthBasedRefinement(knowledge_base=kb), quality_func=F1(),
                 pretrained_model_path=args.pretrained_drill_avg_path, verbose=args.verbose)


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
        title='Class Expression Learning DRILL',
        description='Click Random Examples & Submit.').launch(server_name=server_name,
                                                              server_port=server_port,
                                                              share=False)
def ClosedWorld_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    assert isinstance(onto, OWLOntology_Owlready2)
    return OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=OWLReasoner_Owlready2_TempClasses(ontology=onto),
                                               negation_default=True)


def run(args):
    # Load data
    kb = KnowledgeBase(path=args.path_knowledge_base, reasoner_factory=ClosedWorld_ReasonerFactory)
    # Load model
    drill = load_model(kb, args)
    # Load model
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
