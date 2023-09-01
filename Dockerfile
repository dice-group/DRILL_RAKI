FROM python:3.9
RUN apt-get update && apt-get install -y unzip wget
# set a directory.
WORKDIR /data
# Clone and unzip data
RUN git clone https://github.com/dice-group/DRILL_RAKI
RUN cd DRILL_RAKI && unzip KGs.zip && unzip embeddings.zip && unzip LPs.zip && unzip pre_trained_agents.zip
# Install dependencies
RUN cd DRILL_RAKI && wget --no-check-certificate --content-disposition https://github.com/dice-group/Ontolearn/archive/refs/tags/0.5.1.zip && unzip Ontolearn-0.5.1.zip && cd Ontolearn-0.5.1 && pip install -e .
RUN pip3 install flask==2.2.5
RUN pip3 install gradio==3.41.2

EXPOSE 7860/predict
CMD python /data/DRILL_RAKI/deploy.py --path_knowledge_base "/data/DRILL_RAKI/KGs/$KG" --path_knowledge_base_embeddings "/data/DRILL_RAKI/embeddings/$EMBEDDINGS" --pretrained_drill_avg_path "/data/DRILL_RAKI/pre_trained_agents/$PRE_TRAINED_AGENT" --only_end_point "$INTERFACE"