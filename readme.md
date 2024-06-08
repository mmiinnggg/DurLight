# Introduction
DurLight, an innovative RL-based solution tailored for adaptive traffic signal control, which empowers agent autonomy by facilitating flexible adjustments in action duration and dynamic signal phase invocation.
# Environment & Dependency:
- This code has been tested on Python 3.7, and compatibility with other versions is not guaranteed. It is recommended to use Python versions 3.5 and above.
- For installing CityFlow, it is recommended to follow the instructions provided at https://cityflow.readthedocs.io/en/latest/install.html.
  
|Name| Version |
|---|---------|
|Keras| v2.3.1   |
|tensorflow-gpu| 1.14.0  |
|CityFlow| 1.1.0   |
| tqdm | 4.65.0 |
| numpy | 1.21.5  |
| matplotlib |  3.5.3  |



# Files
* ``runexp.py``
  The main file of experiments where the args can be changed.

 > The **arg '--dataset'** requires special attention, and it should be consistent with the dataset being used. For example, the datasets corresponding to road networks such as Jinan, Hangzhou, and Syn3x3 should have names that respectively match '--dataset==jinan', '--dataset==hangzhou', and '--dataset==3x3'.
 
* ``agent.py``
  Implement RL agent for proposed method.

* ``cityflow_env.py``
  Define a simulator environment to interact with the simulator and obtain needed data.

* ``utility.py``
  Other functions for experiments.

* ``metric/travel_time.py`` & ``metric/throughput.py``
  Two representative measures as evaluation criteria to digitally assess the performance of the method.

* ``data.zip``
   Containing all the used traffic file and road networks datasets. When extracting the 'data.zip' file, the resulting files will be stored in the 'project dir/data' directory.



# How to run
In terminal:
```shell
cd project_dir
```
and then:
```shell
python3 runexp.py
```
