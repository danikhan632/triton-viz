#!/bin/bash
cd frontend
npm run build
rm -rf ../triton_viz/frontend_build
cp -R ./build ../triton_viz/frontend_build
cd ..


