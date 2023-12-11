# RLlib2-eigen

Using eigen for implementing extra rllib2 algorithms.


# Install

First, install the gdyn library from https://github.com/HerveFrezza-Buet/gdyn
First, install the rllib2 library from https://gitlab.inria.fr:biscuit/rllib2/rllib2

Then go to the rllib2/rllib2-eigen directory

```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr
make -j
sudo make install
```