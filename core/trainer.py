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

class Trainer:
    def __init__(self, args):
        self.args = args

    def save_config(self, path):
        with open(path + '/configuration.json', 'w') as file_descriptor:
            temp = vars(self.args)
            json.dump(temp, file_descriptor)

    def start(self):
    print("start parsing ... "+ str(self.args.path_knowledge_base)+" reasoner"+str(ClosedWorld_ReasonerFactory))
        # 1. Parse KG.
        kb = KnowledgeBase(path=self.args.path_knowledge_base, reasoner_factory=ClosedWorld_ReasonerFactory)
    print("done parsing")
        min_num_instances = self.args.min_num_instances_ratio_per_concept * kb.individuals_count()
        max_num_instances = self.args.max_num_instances_ratio_per_concept * kb.individuals_count()

        # 2. Generate Learning Problems.
        print("Learning Problem Generator...")
        lp = LearningProblemGenerator(knowledge_base=kb,
                                      min_length=self.args.min_length,
                                      max_length=self.args.max_length,
                                      min_num_instances=min_num_instances,
                                      max_num_instances=max_num_instances)
        print("Generate balanced learning problems...")
        balanced_examples = lp.get_balanced_n_samples_per_examples(
            n=self.args.num_of_randomly_created_problems_per_concept,
            min_length=self.args.min_length,
            max_length=self.args.max_length,
            min_num_problems=self.args.min_num_concepts,
            num_diff_runs=self.args.min_num_concepts // 2)
        print("initialize DRILL...")
        drill = Drill(knowledge_base=kb, path_of_embeddings=self.args.path_knowledge_base_embeddings,
                      refinement_operator=ModifiedCELOERefinement(
                          knowledge_base=kb) if self.args.refinement_operator == 'ModifiedCELOERefinement' else LengthBasedRefinement(
                          knowledge_base=kb),
                      quality_func=F1(),
                      reward_func=Reward(),
                      batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                      pretrained_model_path=self.args.pretrained_drill_avg_path, verbose=self.args.verbose,
                      max_len_replay_memory=self.args.max_len_replay_memory, epsilon_decay=self.args.epsilon_decay,
                      num_epochs_per_replay=self.args.num_epochs_per_replay,
                      num_episodes_per_replay=self.args.num_episodes_per_replay, learning_rate=self.args.learning_rate,
                      num_of_sequential_actions=self.args.num_of_sequential_actions, num_episode=self.args.num_episode)

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
