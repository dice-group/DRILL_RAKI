from owlapy.model import OWLOntology, OWLReasoner
from owlapy.owlready2 import OWLOntology_Owlready2
from owlapy.owlready2.temp_classes import OWLReasoner_Owlready2_TempClasses
from owlapy.fast_instance_checker import OWLReasoner_FastInstanceChecker
from ontolearn.utils import log_config
import os
import datetime
import logging

def ClosedWorld_ReasonerFactory(onto: OWLOntology) -> OWLReasoner:
    assert isinstance(onto, OWLOntology_Owlready2)
    base_reasoner = OWLReasoner_Owlready2_TempClasses(ontology=onto)
    reasoner = OWLReasoner_FastInstanceChecker(ontology=onto,
                                               base_reasoner=base_reasoner,
                                               negation_default=True)
    return reasoner


def create_experiment_folder(folder_name='Log'):
    if log_config.log_dirs:
        path_of_folder = log_config.log_dirs[-1]
    else:
        directory = os.getcwd() + '/' + folder_name + '/'
        folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path_of_folder = directory + folder_name
        os.makedirs(path_of_folder)
    return path_of_folder, path_of_folder[:path_of_folder.rfind('/')]

def create_logger(*, name, p):
    """
    @todos We should create a better logging.
    @param name:
    @param p:
    @return:
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(p + '/info.log', 'w', 'utf-8')
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    # logger.addHandler(ch) # do not print in console.
    logger.addHandler(fh)

    return logger