# Experiments

dataset for pixel classification model 
```https://drive.google.com/drive/folders/1FpcZvUGVWG_oz1s_gnlc9X82hx6qaIXA?usp=sharing```

Some experiments require additional libraries.

To run experiments with goodpoint detector you'll need:
scikit-learn - pip3 install -U scikit-learn
PyTorch-Ignite - pip install pytorch-ignite

To run get_recipes_from_minecraft_data.py
pip install minecraft_data

To run experiments in test_skills.py you'll need
pip install -U scikit-image

Vision training experiments are deprecated (.pt reading needs to be fixed + torch version)

cliff_v1 - non-visual target-approaching agent (moved to deprecated)

cliff_v2 - visual target-approaching agent. Used in 5_mine_and_craft_with_a_pinch_of_rl_skills.py,
which is available for MalmoPython lib and minecraft 1.11.2. (moved to deprecated)

cliff_v3 - Q-learning based visual target-approaching agent (moved to deprecated)
