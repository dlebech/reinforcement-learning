# Reinforcement Learning

[![Build Status](https://travis-ci.com/dlebech/reinforcement-learning.svg?branch=master)](https://travis-ci.com/dlebech/reinforcement-learning)
[![codecov](https://codecov.io/gh/dlebech/reinforcement-learning/branch/master/graph/badge.svg)](https://codecov.io/gh/dlebech/reinforcement-learning)

Yet another repo with reinforcement learning algorithms :-)

If you want well-implemented algorithms, you're probably better off using the implementatinos in [`keras-rl`](https://github.com/keras-rl/keras-rl), OpenAI's [`baselines`](https://github.com/openai/baselines) or [`stable-baselines`](https://github.com/hill-a/stable-baselines).

I used this code for learning some of the concepts of reinceforcement learning as well as getting more familiar with Tensorflow/Keras such as "manually" updating network weights, calculating gradients and eager execution. As such, the code is not optimized and might not actually work as expected.

Agents:
- Asynchronous Advantage Actor Critic (`a3c`)
- Proximal Policy Optimization (`ppo`)
- Deep Q learning (`dqn`)
- Random (`random`)

`dqn` and `ppo` only support discrete action environments currently.

All agents are built around solving an [OpenAI Gym](https://gym.openai.com/) environment. Currently, the only reliably solvable environment is the `CartPole-v0` (and `v1`). I have not had much luck with the continuous action environments such as `Pendulum-v0` or `MountainCar-v0`.


## Train and play

View options:

```
python main.py -h
```

For training, choose an algorithm and environment and use the `--train`
argument, for example:

```
python main.py --algorithm a3c --save-dir ./output --env-name "CartPole-v1" --train
```

After training play some test episodes by simply removing the `--train` parameter:

```
python main.py --algorithm a3c --save-dir ./output --env-name "CartPole-v1"
```

**Note**: The output model is named after the algorithm and environment, so if you want to train multiple agents for the same algorithm/environment combination, use different output directories.

**Note**: For obvious reasons, the `random` agent cannot really be trained and the `--train` parameter just plays the game without rendering.

## Acknowledgements

- [Raymond Yuan](https://raymond-yuan.github.io/personal-site/) for a nice walkthrough and implementation of [the A3C algorithm](https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296)
  - [Code](https://github.com/tensorflow/models/tree/master/research/a3c_blogpost)
- Arthur Juliani for a nice series of [RL blog posts](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
  - [Code](https://github.com/awjuliani/DeepRL-Agents)


## License

MIT License

Parts of the code for the a3c and random algorithms are copied from Apache
2.0 licensed code with the following notice:
```
Copyright 2016, The TensorFlow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
