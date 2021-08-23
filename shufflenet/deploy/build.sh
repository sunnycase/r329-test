#!/bin/bash
../../sdk/toolchain/bin/aarch64-openwrt-linux-g++ main.cpp -L../../sdk/lib  -lm -lpthread -lstdc++ -laipudrv -o build/shufflenet
cp -r ../../sdk/lib build
cp ../test/input.bin build
patchelf --set-rpath "\$ORIGIN/lib" build/shufflenet