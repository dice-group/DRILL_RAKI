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

from owlapy.model import OWLOntology, OWLReasoner
from owlapy.model import OWLNamedIndividual
from owlapy.model import IRI


import logging
setup_logging()
logger = logging.getLogger(__name__)

# @ TODO: We may want to provide an endpoint without threading.
kb = None
drill = None
args = None
lock = threading.Lock()
loading: bool = False
ready: bool = False

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return update_wrapper(no_cache, view)
def create_flask_app():
    app = Flask(__name__, instance_relative_config=True, )

    @app.route('/concept_learning', methods=['POST'])
    def concept_learning_endpoint():
        """
        Accepts a json objects with parameters "positives" and "negatives". Those must have as value a list of entity
        strings each. Additionally a HTTP form parameter `no_of_hypotheses` can be provided. If not provided, it
        defaults to 1.
        """
        global lock
        global ready
        global args
        lock.acquire()
        try:
            global drill
            global kb
            ready = False
            learning_problem = request.get_json(force=True)
            app.logger.debug(learning_problem)
            no_of_hypotheses = request.form.get("no_of_hypotheses", 1, type=int)
            try:
                from owlapy.model import IRI
                typed_pos = set(map(OWLNamedIndividual, map(IRI.create, set(learning_problem["positives"]))))
                typed_neg = set(map(OWLNamedIndividual, map(IRI.create, set(learning_problem["negatives"]))))
                drill.fit(typed_pos, typed_neg,
                          max_runtime=args.max_test_time_per_concept)
            except Exception as e:
                app.logger.debug(e)
                abort(400)
            import tempfile
            tmp = tempfile.NamedTemporaryFile()
            try:
                drill.save_best_hypothesis(no_of_hypotheses, tmp.name)
            except Exception as ex:
                print(ex)
            hypotheses_ser = io.open(tmp.name+'.owl', mode="r", encoding="utf-8").read()
            from pathlib import Path
            Path(tmp.name+'.owl').unlink(True)
            return Response(hypotheses_ser, mimetype="application/rdf+xml")
        finally:
            ready = True
            lock.release()

    @app.route('/status')
    @nocache
    def status_endpoint():
        global loading
        global ready
        if loading:
            flag = "loading"
        elif ready:
            flag = "ready"
        else:
            flag = "busy"
        status = {"status": flag}
        return status

    @app.before_first_request
    def set_ready():
        global lock
        with lock:
            global loading
            loading = False
            global ready
            ready = True

    return app

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
                        default='pre_trained/DrillHeuristic_averaging.pth', help='Provide a path of .pth file')
    # NN related
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=int, default=.01)
    parser.add_argument("--drill_first_out_channels", type=int, default=32)

    # Concept Learning Testing
    parser.add_argument("--iter_bound", type=int, default=10_000, help='iter_bound during testing.')
    parser.add_argument('--max_test_time_per_concept', type=int, default=3, help='Max. runtime during testing')

    loading = True
    args = parser.parse_args()

    kb = KnowledgeBase(path=args.path_knowledge_base, reasoner_factory=ClosedWorld_ReasonerFactory)
    drill = Drill(knowledge_base=kb, path_of_embeddings=args.path_knowledge_base_embeddings,
                  refinement_operator=LengthBasedRefinement(knowledge_base=kb), quality_func=F1(),
                  batch_size=args.batch_size, num_workers=args.num_workers,
                  pretrained_model_path=args.pretrained_drill_avg_path, verbose=args.verbose,
                  num_of_sequential_actions=args.num_of_sequential_actions)
    app = create_flask_app()
    app.run(host="0.0.0.0", port=9080, processes=1)  # processes=1 is important to avoid copying the kb
