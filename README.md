# Deep Reinforcement Learning for Refinement Operators in ALC

This open-source project contains the Pytorch implementation of DRILL, training and evaluation scripts. 
To foster further reproducible research and alleviate hardware requirements to reproduce the reported results, we provide pretrained models on all datasets.

# Install Ontolearn
wget --no-check-certificate --content-disposition https://github.com/dice-group/Ontolearn/archive/refs/tags/v.0.0.1.zip
unzip Ontolearn-v.0.0.1.zip
cd Ontolearn-v.0.0.1
conda create --name temp python=3.8
conda activate temp
pip install -e .

# Installation
Create a anaconda virtual environment and install dependencies.
```
git clone https://github.com/dice-group/DRILL
# Create anaconda virtual enviroment
conda create -n drill_env python=3.9
# Active virtual enviroment 
conda activate drill_env
cd DRILL
# Install our developed framework. It may take few minutes
pip install -e .
# For the Endpoint
pip install flask==2.1.2
# Test the installation. No error should occur.
python -c "import ontolearn"
```
# Preprocessing 
Unzip knowledge graphs, embeddings, learning problems and pretrained models.
```
unzip KGs.zip
unzip embeddings.zip
unzip pre_trained_agents.zip
unzip LPs.zip
```

# Training

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

### Run Endpoint
TODO:
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
