from ontolearn.knowledge_base import KnowledgeBase
from .static_funcs import ClosedWorld_ReasonerFactory
import ontolearn
from ontolearn.learning_problem_generator import LearningProblemGenerator
from ontolearn.refinement_operators import LengthBasedRefinement, ModifiedCELOERefinement
from owlapy.render import DLSyntaxObjectRenderer
from core import helper_classes
from core.model import Drill
from ontolearn.metrics import F1
from ontolearn.heuristics import Reward
import time
from collections import defaultdict
import copy
from itertools import chain, tee
import random
from typing import DefaultDict, Dict, Set, Optional, Iterable, List, Type, Final, Generator
from ontolearn.value_splitter import AbstractValueSplitter, BinningValueSplitter
from owlapy.model.providers import OWLDatatypeMaxInclusiveRestriction, OWLDatatypeMinInclusiveRestriction
from owlapy.vocab import OWLFacet

from ontolearn.abstracts import BaseRefinement
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.model import OWLObjectPropertyExpression, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, \
    OWLObjectIntersectionOf, OWLClassExpression, OWLNothing, OWLThing, OWLNaryBooleanClassExpression, \
    OWLObjectUnionOf, OWLClass, OWLObjectComplementOf, OWLObjectMaxCardinality, OWLObjectMinCardinality, \
    OWLDataSomeValuesFrom, OWLDatatypeRestriction, OWLLiteral, OWLObjectInverseOf, OWLDataProperty, \
    OWLDataHasValue, OWLDataPropertyExpression
from ontolearn.search import Node, OENode


class ScalableLengthBasedRefinement(ontolearn.refinement_operators.BaseRefinement):
    """ A top down refinement operator refinement operator in ALC."""

    def __init__(self, knowledge_base: KnowledgeBase):
        super().__init__(knowledge_base)
        # 1. Number of named classes and sanity checking
        self.max_len_refinement_top = 1
        self.top_refinements = []
        for ref in self.refine_top():
            self.top_refinements.append(ref)

    def refine_top(self) -> Iterable:
        """ Refine Top Class Expression """
        """ (1) Store all named classes """
        iterable_container = []
        all_subs = [i for i in self.kb.get_all_sub_concepts(self.kb.thing)]
        iterable_container.append(all_subs)
        """ (2) Negate (1) and store it """
        iterable_container.append(self.kb.negation_from_iterables((i for i in all_subs)))
        """ (3) Add Nothing """
        iterable_container.append([self.kb.nothing])
        """ (4) Get all most general restrictions and store them forall r. T, \\exist r. T """
        iterable_container.append(self.kb.most_general_universal_restrictions(domain=self.kb.thing, filler=None))
        iterable_container.append(self.kb.most_general_existential_restrictions(domain=self.kb.thing, filler=None))
        """ (5) Generate all refinements of given concept that have length less or equal to the maximum refinement
         length constraint """
        yield from self.apply_union_and_intersection_from_iterable(iterable_container)

    def apply_union_and_intersection_from_iterable(self, cont: Iterable[Generator]) -> Iterable:
        """ Create Union and Intersection OWL Class Expressions
        1. Create OWLObjectIntersectionOf via logical conjunction of cartesian product of input owl class expressions
        2. Create OWLObjectUnionOf class expression via logical disjunction pf cartesian product of input owl class
         expressions
        Repeat 1 and 2 until all concepts having max_len_refinement_top reached.
        """
        cumulative_refinements = dict()
        """ 1. Flatten list of generators """
        for class_expression in chain.from_iterable(cont):
            if class_expression is not self.kb.nothing:
                """ 1.2. Store qualifying concepts based on their lengths """
                cumulative_refinements.setdefault(self.len(class_expression), set()).add(class_expression)
            else:
                """ No need to union or intersect Nothing, i.e. ignore concept that does not satisfy constraint"""
                yield class_expression
        """ 2. Lengths of qualifying concepts """
        lengths = [i for i in cumulative_refinements.keys()]

        seen = set()
        larger_cumulative_refinements = dict()
        """ 3. Iterative over lengths """
        for i in lengths:  # type: int
            """ 3.1 Return all class expressions having the length i """
            yield from cumulative_refinements[i]
            """ 3.2 Create intersection and union of class expressions having the length i with class expressions in
             cumulative_refinements """
            for j in lengths:
                """ 3.3 Ignore if we have already createdValid intersection and union """
                if (i, j) in seen or (j, i) in seen:
                    continue

                seen.add((i, j))
                seen.add((j, i))

                len_ = i + j + 1

                if len_ <= self.max_len_refinement_top:
                    """ 3.4 Intersect concepts having length i with concepts having length j"""
                    intersect_of_concepts = self.kb.intersect_from_iterables(cumulative_refinements[i],
                                                                             cumulative_refinements[j])
                    """ 3.4 Union concepts having length i with concepts having length j"""
                    union_of_concepts = self.kb.union_from_iterables(cumulative_refinements[i],
                                                                     cumulative_refinements[j])
                    res = set(chain(intersect_of_concepts, union_of_concepts))

                    # Store newly generated concepts at 3.2.
                    if len_ in cumulative_refinements:
                        x = cumulative_refinements[len_]
                        cumulative_refinements[len_] = x.union(res)
                    else:
                        if len_ in larger_cumulative_refinements:
                            x = larger_cumulative_refinements[len_]
                            larger_cumulative_refinements[len_] = x.union(res)
                        else:
                            larger_cumulative_refinements[len_] = res

        for k, v in larger_cumulative_refinements.items():
            yield from v

    def refine_atomic_concept(self, class_expression: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """
        Refine an atomic class expressions, i.e,. length 1
        """
        assert isinstance(class_expression, OWLClassExpression)
        for i in self.top_refinements:
            # No need => Daughter ⊓ Daughter
            # No need => Daughter ⊓ \bottom
            if i.is_owl_nothing() is False and (i != class_expression):
                yield self.kb.intersection((class_expression, i))
        # Previously; yield self.kb.intersection(node.concept, self.kb.thing)

    def refine_complement_of(self, class_expression: OWLObjectComplementOf) -> Iterable[OWLClassExpression]:
        """
        Refine OWLObjectComplementOf
        1- Get All direct parents
        2- Negate (1)
        3- Intersection with T
        """
        assert isinstance(class_expression, OWLObjectComplementOf)
        yield from self.kb.negation_from_iterables(self.kb.get_direct_parents(self.kb.negation(class_expression)))
        yield self.kb.intersection((class_expression, self.kb.thing))

    def refine_object_some_values_from(self, class_expression: OWLObjectSomeValuesFrom) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLObjectSomeValuesFrom)
        # rule 1: \exists r.D = > for all r.E
        for i in self.refine(class_expression.get_filler()):
            yield self.kb.existential_restriction(i, class_expression.get_property())
        # rule 2: \exists r.D = > \exists r.D AND T
        yield self.kb.intersection((class_expression, self.kb.thing))

    def refine_object_all_values_from(self, class_expression: OWLObjectAllValuesFrom) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLObjectAllValuesFrom)
        # rule 1: \forall r.D = > \forall r.E
        for i in self.refine(class_expression.get_filler()):
            yield self.kb.universal_restriction(i, class_expression.get_property())
        # rule 2: \forall r.D = > \forall r.D AND T
        yield self.kb.intersection((class_expression, self.kb.thing))

    def refine_object_union_of(self, class_expression: OWLObjectUnionOf) -> Iterable[OWLClassExpression]:
        """
        Refine C =A AND B
        """
        assert isinstance(class_expression, OWLObjectUnionOf)
        operands: List[OWLClassExpression] = list(class_expression.operands())
        for i in operands:
            for ref_concept_A in self.refine(i):
                if ref_concept_A == class_expression:
                    # No need => Person OR MALE => rho(Person) OR MALE => MALE OR MALE
                    yield class_expression
                yield self.kb.union((class_expression, ref_concept_A))

    def refine_object_intersection_of(self, class_expression: OWLClassExpression) -> Iterable[OWLClassExpression]:
        """
        Refine C =A AND B
        """
        assert isinstance(class_expression, OWLObjectIntersectionOf)
        operands: List[OWLClassExpression] = list(class_expression.operands())
        for i in operands:
            for ref_concept_A in self.refine(i):
                if ref_concept_A == class_expression:
                    # No need => Person ⊓ MALE => rho(Person) ⊓ MALE => MALE ⊓ MALE
                    yield class_expression
                # TODO: No need to intersect disjoint expressions
                yield self.kb.intersection((class_expression, ref_concept_A))

    def refine(self, class_expression) -> Iterable[OWLClassExpression]:
        assert isinstance(class_expression, OWLClassExpression)
        if class_expression.is_owl_thing():
            yield from self.top_refinements
        elif class_expression.is_owl_nothing():
            yield from {class_expression}
        elif self.len(class_expression) == 1:
            yield from self.refine_atomic_concept(class_expression)
        elif isinstance(class_expression, OWLObjectComplementOf):
            yield from self.refine_complement_of(class_expression)
        elif isinstance(class_expression, OWLObjectSomeValuesFrom):
            yield from self.refine_object_some_values_from(class_expression)
        elif isinstance(class_expression, OWLObjectAllValuesFrom):
            yield from self.refine_object_all_values_from(class_expression)
        elif isinstance(class_expression, OWLObjectUnionOf):
            yield from self.refine_object_union_of(class_expression)
        elif isinstance(class_expression, OWLObjectIntersectionOf):
            yield from self.refine_object_intersection_of(class_expression)
        else:
            raise ValueError


class Trainer:
    def __init__(self, args):
        self.args = args

    def save_config(self, path):
        with open(path + '/configuration.json', 'w') as file_descriptor:
            temp = vars(self.args)
            json.dump(temp, file_descriptor)

    def start(self):
        # 1. Parse KG.
        start_time = time.time()
        print('Reading input data.....')
        kb = KnowledgeBase(path=self.args.path_knowledge_base, reasoner_factory=ClosedWorld_ReasonerFactory)
        print(f'Reading input data took:{time.time() - start_time}')
        max_num_instances = self.args.max_num_instances_ratio_per_concept * kb.individuals_count() if self.args.max_num_instances_ratio_per_concept is not None else kb.individuals_count()
        min_num_instances = self.args.min_num_instances_ratio_per_concept * kb.individuals_count() if self.args.min_num_instances_ratio_per_concept is not None else 1

        print('Generating Learning Problems...')
        start_time = time.time()
        # 2. Generate Learning Problems... Very time time-consuming (didn't terminate within 20 minutes
        # Number of named classes: 3593 etc.)
        # This stems from the fact that,
        # LengthBasedRefinement stores refinements of Top concept until the length of 4,e.g.
        # to generate only length 2 refinements we store (3593 * 3593) concepts
        # A workaround introduced by
        total_named_concepts = len(set(i for i in kb.ontology().classes_in_signature()))
        if total_named_concepts < 50:
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
        else:
            balanced_examples = []
        print(f'Generating Learning Problems took:{time.time() - start_time}')

        start_time = time.time()
        if total_named_concepts < 50:
            refinement_operator = ModifiedCELOERefinement(
                knowledge_base=kb) if self.args.refinement_operator == 'ModifiedCELOERefinement' else LengthBasedRefinement(
                knowledge_base=kb)
        else:
            refinement_operator = ModifiedCELOERefinement(
                knowledge_base=kb) if self.args.refinement_operator == 'ModifiedCELOERefinement' else ScalableLengthBasedRefinement(
                knowledge_base=kb)

        print('Initializing DRILL...')
        drill = Drill(knowledge_base=kb, path_of_embeddings=self.args.path_knowledge_base_embeddings,
                      refinement_operator=refinement_operator,
                      quality_func=F1(),
                      reward_func=Reward(),
                      batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                      pretrained_model_path=self.args.pretrained_drill_avg_path, verbose=self.args.verbose,
                      max_len_replay_memory=self.args.max_len_replay_memory, epsilon_decay=self.args.epsilon_decay,
                      num_epochs_per_replay=self.args.num_epochs_per_replay,
                      num_episodes_per_replay=self.args.num_episodes_per_replay, learning_rate=self.args.learning_rate,
                      num_of_sequential_actions=self.args.num_of_sequential_actions, num_episode=self.args.num_episode)
        print(f'Initializing DRILL took:{time.time() - start_time}')

        drill.train(balanced_examples)
        drill.save_weights()
        renderer = DLSyntaxObjectRenderer()
        test_data = [
            {'target_concept': renderer.render(rl_state.concept),
             'positive_examples': {i.get_iri().as_str() for i in typed_p},
             'negative_examples': {i.get_iri().as_str() for i in typed_n}, 'ignore_concepts': set()} for
            rl_state, typed_p, typed_n in balanced_examples]
        for result_dict, learning_problem in zip(
                drill.fit_from_iterable(test_data, max_runtime=self.args.max_test_time_per_concept),
                test_data):
            print(f'\nTarget Class Expression:{learning_problem["target_concept"]}')
            print(
                f'| sampled E^+|:{len(learning_problem["positive_examples"])}\t| sampled E^-|:{len(learning_problem["negative_examples"])}')
            for k, v in result_dict.items():
                print(f'{k}:{v}')
