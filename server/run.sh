#!/bin/bash

chmod +x llava-v1.5-7b-q4.llamafile

./llava-v1.5-7b-q4.llamafile --server --nobrowser &
python server.py && fg
