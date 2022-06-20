import os
from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# Paths
cfg.data_folder = ''
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.path_knowledge_base = "KGs/Family/family-benchmark_rich_background.owl"
cfg.path_knowledge_base_embeddings = "embeddings/ConEx_Family/ConEx_entity_embeddings.csv"
cfg.pretrained_drill_avg_path = None  # if we want to retrain a pretrained-drill

# Computation
cfg.num_workers = 1

cfg.seed = 0
cfg.verbose = 10

cfg.min_num_concepts = 1
cfg.min_length = 4
cfg.max_length = 5

cfg.min_num_instances_ratio_per_concept = .01
cfg.max_num_instances_ratio_per_concept = .6
cfg.num_of_randomly_created_problems_per_concept = 1

# Available refinement opts: ['ModifiedCELOERefinement', 'LengthBasedRefinement']
cfg.refinement_operator = 'LengthBasedRefinement'

# DQL related
cfg.gamma = .99
cfg.num_episode = 5
cfg.epsilon_decay = .01
cfg.max_len_replay_memory = 1024

cfg.num_epochs_per_replay = 3
cfg.num_episodes_per_replay = 10
cfg.num_of_sequential_actions = 3
cfg.relearn_ratio = 1
cfg.use_illustrations = True
cfg.use_target_net = False

cfg.batch_size = 512
cfg.learning_rate = .01
cfg.drill_first_out_channels = 32

cfg.iter_bound = 10_00
cfg.max_test_time_per_concept = 3

basic_cfg = cfg
