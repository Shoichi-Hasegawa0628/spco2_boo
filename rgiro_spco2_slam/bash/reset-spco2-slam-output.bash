#!/bin/bash

rm -r ../data/output/test/*
rm -r ../src/*.txt
rm -r ../src/*.tar
rm -r ../src/*.npy
rm -r ../src/wideresnet.py

mkdir -p ../data/output/test/img/
mkdir -p ../data/output/test/map/
mkdir -p ../data/output/test/particle/
mkdir -p ../data/output/test/tmp/
mkdir -p ../data/output/test/weight/
