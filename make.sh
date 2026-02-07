#!/bin/bash

quarto render *.qmd

rm -rf outputs
mkdir outputs
mv *.html outputs
