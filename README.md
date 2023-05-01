# RLlib2

This is an upgrade of the former rlib project (see https://github.com/HerveFrezza-Buet/RLlib).

It is based on C++20 features.


# Install

First, install the gdyn library from https://github.com/HerveFrezza-Buet/gdyn

Then go to the rllib2 directory you have git-cloned.

```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr
make -j
sudo make install
```