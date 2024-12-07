#!/bin/bash
cd frontend
npm run build
rm -rf ../triton_viz/build
cp -R ./build ../triton_viz/build
cd ..


