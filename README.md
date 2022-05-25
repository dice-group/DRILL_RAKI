# Deep Reinforcement Learning for Refinement Operators in ALC

This open-source project supported by [RAKI](https://raki-projekt.de/)  contains the Pytorch implementation of DRILL, training and evaluation scripts.
DRILL is a convolutional deep reinforcement learning agent that effectively accelerates the class expression learning process.
To this end, DRILL assigns higher scores to those class expression that are likely to lead the search towards goal expressions.

# Installation
Create a anaconda virtual environment and install dependencies.
```
# Clone the repository and create a python virtual enviroment via anaconda
git clone https://github.com/dice-group/DRILL_RAKI && conda create -n drill_env python=3.9 && conda activate drill_env
# Install requirements
cd DRILL_RAKI && wget --no-check-certificate --content-disposition https://github.com/dice-group/Ontolearn/archive/refs/tags/0.5.1.zip
unzip Ontolearn-0.5.1.zip && cd Ontolearn-0.5.1 && pip install -e . && cd ..
# For the Endpoint only
pip install flask==2.1.2
# Test the installation. No error should occur.
python -c "import ontolearn"
```
# Preprocessing 
Unzip knowledge graphs, embeddings, learning problems and pretrained models.
```
unzip KGs.zip && unzip embeddings.zip && unzip LPs.zip && unzip pre_trained_agents.zip
```
# Training Deep Reinforcement Learning Agent for Class Expression Learning
## Knowledge Graph Embeddings
#### Install dice-embeddings framework
Install our framework to learn vector representations for knowledge graphs
```
git clone https://github.com/dice-group/dice-embeddings.git && pip install -r dice-embeddings/requirements.txt
```
Convert an OWL knowledge base into ntriples to create training dataset for KGE.
```python
import rdflib
g = rdflib.Graph()
g.parse("KGs/Family/family-benchmark_rich_background.owl")
g.serialize("KGs/Family/train.txt", format="nt")
```
#### Compute Embeddings
Executing the following command results in creating a folder (KGE_Embeddings) containing all necessary information about the KGE process.
```
python dice-embeddings/main.py --path_dataset_folder "KGs/Family" --storage_path "KGE_Embeddings" --model "ConEx"
```
## Train DRILL
To train DRILL, we need to provide the path of a knowledgebase (KGs/Biopax/biopax.owl) and embeddings
```
python drill_train.py --path_knowledge_base "KGs/Family/family-benchmark_rich_background.owl" --path_knowledge_base_embeddings "KGE_Embeddings/2022-05-25 14:22:34.353957/ConEx_entity_embeddings.csv" --num_episode 2 --min_num_concepts 2 --num_of_randomly_created_problems_per_concept 1 --relearn_ratio 2
```

### Run an endpoint for DRILL
To use the endpoint for a pretrained agent, provide the path of the knowledge base as well as the pretrained agent.
```
python flask_end_point.py --pretrained_drill_avg_path "Log/20220525_142307_565226/DrillHeuristic_averaging.pth" --path_knowledge_base "KGs/Family/family-benchmark_rich_background.owl" --path_knowledge_base_embeddings "KGE_Embeddings/2022-05-25 14:22:34.353957/ConEx_entity_embeddings.csv"
```
### Standard Class Expression Learning
```
curl -X POST http://0.0.0.0:9080/concept_learning -H 'Content-Type: application/json' -d '{"positives": ["http://www.benchmark.org/family#F9M149"],"negatives": [ "http://www.benchmark.org/family#F9F169"]}'
# Expected output
<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xml:base="https://dice-research.org/predictions/1653391568.558438"
         xmlns="https://dice-research.org/predictions/1653391568.558438#">

<owl:Ontology rdf:about="https://dice-research.org/predictions/1653391568.558438">
  <owl:imports rdf:resource="file://KGs/Family/family-benchmark_rich_background.owl"/>
</owl:Ontology>
<owl:AnnotationProperty rdf:about="#f1_score"/>
<owl:Class rdf:about="#Pred_0">
  <owl:equivalentClass rdf:resource="http://www.benchmark.org/family#Male"/>
  <f1_score rdf:datatype="http://www.w3.org/2001/XMLSchema#decimal">1.0</f1_score>
</owl:Class>
</rdf:RDF>
```
### Positive Only Class Expression Learning
```
curl -X POST http://0.0.0.0:9080/concept_learning -H 'Content-Type: application/json' -d '{"positives": ["http://www.benchmark.org/family#F9M149"],"negatives": []}'
# Note that negatives must be an empty list 
# Expected output
<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xml:base="https://dice-research.org/predictions/1653398295.3083727"
         xmlns="https://dice-research.org/predictions/1653398295.3083727#">

<owl:Ontology rdf:about="https://dice-research.org/predictions/1653398295.3083727">
  <owl:imports rdf:resource="file://KGs/Family/family-benchmark_rich_background.owl"/>
</owl:Ontology>

<owl:AnnotationProperty rdf:about="#f1_score"/>
<owl:Class rdf:about="#Pred_0">
  <owl:equivalentClass>
    <owl:Class>
      <owl:complementOf rdf:resource="http://www.benchmark.org/family#Father"/>
    </owl:Class>
  </owl:equivalentClass>
  <f1_score rdf:datatype="http://www.w3.org/2001/XMLSchema#decimal">1.0</f1_score>
</owl:Class>
</rdf:RDF>
```
## Prior Knowledge Injection
TODO: Add "ignore_concepts": ["Male","Father","Son","Brother","Grandfather","Grandson"]


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
