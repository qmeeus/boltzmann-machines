#!/bin/sh

case "$1" in
"test")
    exec make test;;
"notebook")
    exec jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser;;
"bash")
    exec bash;;
*)
    #exec tensorboard --logdir ./outputs/logs > /dev/null 2>&1 &
    exec python -m boltzmann_machines.mains."$@";;
esac
