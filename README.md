# CrowdNav_Cai

<p align="center">
<img src="https://github.com/user-attachments/assets/b914c1ef-27dd-4833-977b-a062cd415919"  width="300" height="300"/>
</p>

this repo makes for training and testing a Crowd navigation or a map based navigation

you can run online reinforcement learning in a navigation task

## Simulator rendering

1. the pink can not be seen by 2d scan, the blue can be seen by 2d scan. (this is for CrowdNav testing or training)

2. A pink line is a path which is made by Dijkstra algorithm, and A pink triangle is a subgoal, A Yellow triangle is a goal.

3. apply map (with walls)

4. only walls (on the other way, only map is applied)


## How to install 

os: ubuntu20.04 , python: 3.8.x


### 1. make path 

```
mkdir catkin_ws && cd catkin_ws && mkdir src && cd src
```

### 2. git clone python package

```
git clone https://github.com/CAI23sbP/CrowdNav_Cai.git && cd CrowdNav_Cai && pip3 install -e . && cd .. 
```

### 3. git clone python sub-packages

#### 3.1. git clone pymap2d & build

```
git clone https://github.com/CAI23sbP/pymap2d.git && pip3 install -e . && cd ..
```

#### 3.2. git clone py-rvo2 <ver: danieldugas-0.0.2> 

```
git clone https://github.com/danieldugas/Python-RVO2.git && cd Python-RVO2 && python3 setup.py build && python3 setup.py install && cd ../..
```

#### 3.3. git clone range_libc

```
git clone https://github.com/CAI23sbP/range_libc.git && cd range_libc/pywrapper && pip3 install -e. && cd ../..
```


## How to train

### 1. set your config

see config file in ``` crowd_nav/configs/*.yaml ```

### 3. training 


```
train.py 
```


## How to test

### 1. testing your model


```
test.py --n_eval <number of eval> --weight_path <weight path '../../.pt','../../.pth', ...> --render <visualize or not> 
```


## How make your agent

### 1. make a env (with a reward and a observation manager)

see detail ```example_scan_sim.py``` (path: crowd_sim/envs/)

### 2. make policy

see detail ```example.py``` (path: crowd_sim/envs/policy/network_policies/)

### 3. make a model (with a feature extractor)

if you want to use libraries which are caled imitation , stable-baslines3 and sb3-contrib, You must make the extractor. 
see detail ```example_extractor.py``` (path: drl_utils/algorithms/extractors/)

### 4. make your config file

see detail ```base_config.yaml``` (path: crowd_nav/configs/)

### 5. make train_<your_model_name>.py

see detail ```train_<model_name>.py```

### 6. add your mode in test.py




## Reference code
[1] [CrowdNav](https://github.com/vita-epfl/CrowdNav)

[2] [NavRep](https://github.com/ethz-asl/navrep)

[3] [CrowdNav_DSRNN](https://github.com/Shuijing725/CrowdNav_DSRNN)

[4] [Pas_CrowdNav](https://github.com/yejimun/PaS_CrowdNav)