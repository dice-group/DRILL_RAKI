"""
Deploy our approach
# positive examples as a string
http://www.biopax.org/examples/glycolysis#complex273,http://www.biopax.org/examples/glycolysis#complex282

http://www.biopax.org/examples/glycolysis#complex241



curl -X POST http://127.0.0.1:7860/predict -H 'Content-Type: application/json' -d '{"positives": ["http://www.biopax.org/examples/glycolysis#complex265"],"negatives": [ "http://www.biopax.org/examples/glycolysis#complex191"]}'


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
from flask import Flask, request, jsonify

app = Flask(__name__)
kb = None
drill = None
renderer = DLSyntaxObjectRenderer()
topk = None
max_test_time_per_concept = None


def collect_lp():
    if "ignore_concepts" in learning_problem:
        concepts_to_ignore = set(
            filter(lambda _: _.get_iri().get_remainder() in learning_problem["ignore_concepts"],
                   kb.ontology().classes_in_signature()))
        if len(concepts_to_ignore) > 0:
            # TODO: Do not ask me why we have this ignore_and_copy() :(
            kb = kb.ignore_and_copy(ignored_classes=concepts_to_ignore)
            drill.knowledge_base = kb
            drill.concepts_to_ignore = concepts_to_ignore


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
        drill.save_best_hypothesis(topk, tmp.name)
    data = []
    for i in drill.best_hypotheses(topk):
        str_rep, f1_score = renderer.render(i.concept), i.quality
        data.append([str_rep, f1_score])

    # Converting to int from int64
    return jsonify({"prediction": data})


@app.route('/', methods=['GET'])
def hello():
    return "<p>Hello, World!</p>"


@app.route('/status', methods=['GET'])
def status():
    return "<p>Hello, World!</p>"


def prepare_learning_problem(positive_examples: str, negative_examples: str, app=None):
    pos_str_list = set(positive_examples.split(","))
    neg_str_list = set(negative_examples.split(","))

    if app:
        app.logger.debug(pos_str_list)
        app.logger.debug(neg_str_list)
    else:
        print("Positive Examples:", pos_str_list)
        print("Negative Examples:", neg_str_list)

    if len(pos_str_list) < 20:
        str_lr = f'E^+:{",".join(pos_str_list)}\nE^-:{",".join(neg_str_list)}\n'
    else:
        str_lr = f'|E^+|:{len(pos_str_list)}\n|E^-|:{len(neg_str_list)}\n'

    # Map str IRI into OWLNamedIndividual
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, pos_str_list)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, neg_str_list)))

    print("Typed Positive Examples:", typed_pos)
    print("Typed Negative Examples:", typed_neg)
    return typed_pos, typed_neg, str_lr


def predict_gradio(positive_examples: str, negative_examples: str, save_predictions: bool):
    typed_pos, typed_neg, str_lr = prepare_learning_problem(positive_examples, negative_examples)
    drill.fit(typed_pos, typed_neg, max_runtime=max_test_time_per_concept)
    if save_predictions:
        tmp = tempfile.NamedTemporaryFile()
        drill.save_best_hypothesis(topk, tmp.name)
    data = []
    for i in drill.best_hypotheses(topk):
        str_rep, f1_score = renderer.render(i.concept), i.quality
        data.append([str_rep, f1_score])
    return str_lr, pd.DataFrame(data=data, columns=["Concept", "F1-score"])

def launch_web_application(server_name: str = None, server_port: int = None):
    gr.Interface(
        fn=predict_gradio,
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
    global kb
    # Load data.
    kb = KnowledgeBase(path=args.path_knowledge_base, reasoner_factory=ClosedWorld_ReasonerFactory)
    global drill
    # Load model.
    drill = Drill(knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
                  refinement_operator=LengthBasedRefinement(knowledge_base=kb), quality_func=F1(),
                  pretrained_model_path=args.pretrained_drill_avg_path)
    global topk
    topk = args.topk
    global max_test_time_per_concept
    max_test_time_per_concept = args.max_test_time_per_concept
    if args.only_end_point == 0:
        launch_web_application(server_name=args.server_name,
                               server_port=args.server_port)
    else:
        app.run(host=args.server_name, port=args.server_port, processes=1)
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_knowledge_base", type=str)
    parser.add_argument("--path_knowledge_base_embeddings", type=str)
    parser.add_argument('--pretrained_drill_avg_path', type=str,
                        default=None,help='Provide a path of .pth file')
    parser.add_argument('--max_test_time_per_concept', type=int, default=3, help='Max. runtime during testing')
    parser.add_argument("--topk", type=int, default=10, help='Return top k concepts')
    parser.add_argument("--only_end_point", type=int, default=1, help="0 for web service, 1 for endpoint")
    parser.add_argument('--server_port', default=7860, type=int)
    parser.add_argument('--server_name', default="0.0.0.0")
    run(parser.parse_args())
