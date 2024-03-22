#!/bin/bash

chmod +x server/llamafile/llava-v1.5-7b-q4.llamafile
chmod +x server/llamafile/mistral-7b-instruct-v0.2.Q5_K_M.llamafile
chmod +x server/llamafile/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile

(
  cd server
  python server.py
) &
(
  cd client
  npm run dev
) && fg
