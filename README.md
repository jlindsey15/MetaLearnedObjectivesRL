Based on https://github.com/louiskirsch/metagenrl

## Installation

Install the following dependencies (in a virtualenv preferably)
```bash
pip3 install ray[tune]==0.7.7 gym[all] mujoco_py>=2 tensorflow-gpu==1.13.2 scipy numpy
```

This code base uses [ray](https://github.com/ray-project/ray), if you would like to use multiple machines,
see [the ray documentation for details](https://ray.readthedocs.io/en/latest/using-ray-on-a-cluster.html).

We also make use of ray's native tensorflow ops. Please compile them by running
```bash
python3 -c 'import ray; from pyarrow import plasma as plasma; plasma.build_plasma_tensorflow_op()'
```

## Meta Training


```bash
python3 ray_experiments.py train --objective_type learned-reinforce-3factor-withstate --reset_after_low 50000 --reset_after_high 500000 --env_name Reacher-v2 --agent_count 20 --timesteps 1000000
```

Example OBJECTIVE_TYPE: learned-withstate-rank1
By default, this requires a local machine with 4 GPUs to run 20 agents in parallel.

## Meta Testing


```bash
python3.6 ray_experiments.py test --objective_type learned-reinforce-3factor-withstate --objective TRAINING_DIRECTORY --chkp -1 --name DESIRED_NAME_FOR_EVALUATION_LOGS --env_name Reacher-v2 --timesteps 500000
```

TRAINING_DIRECTORY will look something like public-None_cd08d900_2021-12-22_02-07-18cep_yk9o

This only requires a single GPU on your machine.


