# Deep Reinforcement Learning for Refinement Operators in ALC

This open-source project supported by [RAKI](https://raki-projekt.de/)  contains the Pytorch implementation of DRILL, training and evaluation scripts.
DRILL is a convolutional deep reinforcement learning agent that effectively accelerates the class expression learning process.
To this end, DRILL assigns higher scores to those class expression that are likely to lead the search towards goal expressions.

# Installation
Create a anaconda virtual environment and install dependencies.
```
git clone https://github.com/dice-group/DRILL_RAKI
# Create anaconda virtual enviroment
conda create -n drill_env python=3.9
# Active virtual enviroment 
conda activate drill_env
cd DRILL_RAKI
wget --no-check-certificate --content-disposition https://github.com/dice-group/Ontolearn/archive/refs/tags/v.0.0.1.zip
unzip Ontolearn-v.0.0.1.zip
cd Ontolearn-v.0.0.1
pip install -e .
# For the Endpoint
pip install flask==2.1.2
# Test the installation. No error should occur.
python -c "import ontolearn"
cd ..
```
# Preprocessing 
Unzip knowledge graphs, embeddings, learning problems and pretrained models.
```
unzip KGs.zip
unzip embeddings.zip
unzip LPs.zip
```

# Training DRILL on Biopax Knowledge Base
Execute the following script to train DRILL on Biopax knowledge base with ConEx Embeddings.
```
# Train DRILL on Biopax and report eval results
python drill_train.py --path_knowledge_base "KGs/Biopax/biopax.owl" --path_knowledge_base_embeddings "embeddings/ConEx_Biopax/ConEx_entity_embeddings.csv" --num_episode 2 --min_num_concepts 2 --num_of_randomly_created_problems_per_concept 1 --relearn_ratio 1
```
As a result of this execution, a log file is created. This log file contains a subfolder for this particular training. Therein, a log file along with pretrained-agent is present.
### Run Endpoint for DRILL
To use the endpoint for a pretrained agent, provide the path of the knowledge base as well as the pretrained agent.
```
python flask_end_point.py --pretrained_drill_avg_path "Log/20220524_125502_665711/DrillHeuristic_averaging.pth" --path_knowledge_base "KGs/Biopax/biopax.owl" --path_knowledge_base_embeddings "embeddings/ConEx_Biopax/ConEx_entity_embeddings.csv"
```
### Send a Request
```
jq '
   .problems
     ."((pathwayStep ⊓ (∀INTERACTION-TYPE.Thing)) ⊔ (sequenceInterval ⊓ (∀ID-VERSION.Thing)))"
   | {
      "positives": .positive_examples,
      "negatives": .negative_examples
     }' LPs/Biopax/lp.json         | curl -d@- http://0.0.0.0:9080/concept_learning
```

## Knowledge Graph Embeddings
#### Install dice-embeddings framework
```
git clone https://github.com/dice-group/dice-embeddings.git
pip install -r dice-embeddings/requirements.txt
mkdir -p dice-embeddings/KGs/Biopax
```
Convert an OWL knowledge base into ntriples to create training dataset for KGE.
```python
import rdflib
g = rdflib.Graph()
g.parse("KGs/Biopax/biopax.owl")
g.serialize("dice-embeddings/KGs/Biopax/train.txt", format="nt")
```
#### Compute Embeddings
Executing the following command results in creating a folder (KGE_Embeddings) containing all necessary information about the KGE process.
```
python dice-embeddings/main.py --path_dataset_folder "dice-embeddings/KGs/Biopax" --storage_path "KGE_Embeddings" --model "ConEx"
```
## Train DRILL
To train DRILL, we need to provide the path of a knowledgebase (KGs/Biopax/biopax.owl) and embeddings
```
python drill_train.py --path_knowledge_base "KGs/Biopax/biopax.owl" --path_knowledge_base_embeddings "KGE_Embeddings/2022-05-13 11:02:53.276242/ConEx_entity_embeddings.csv" --num_episode 2 --min_num_concepts 2 --num_of_randomly_created_problems_per_concept 1 --relearn_ratio 5 --use_illustrations False
```



## How to cite
```
@article{demir2021drill,
  title={DRILL--Deep Reinforcement Learning for Refinement Operators in $$\backslash$mathcal $\{$ALC$\}$ $},
  author={Demir, Caglar and Ngomo, Axel-Cyrille Ngonga},
  journal={arXiv preprint arXiv:2106.15373},
  year={2021}
}
```

For any further questions or suggestions, please contact:  ```caglar.demir@upb.de``` / ```caglardemir8@gmail.com```
