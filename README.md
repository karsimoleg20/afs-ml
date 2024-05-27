# AFS ML

Backend for AFS app (floor cover calculator)

# How to use?

## Environment

### Create

`python3 -m venv venv`

### Activate

`source venv/bin/activate`

## Install dependencies

`pip install -r requirements.txt`

## Train model

`python train_rooms.py`

## Run server with image processing endpoint

`python api.py`

## Deployment

Use Dockerfile for container that can be deployed to any cloud platform.

`docker build -t afs-ml .`

## Codespaces

`docker run -it -d --net=host karsimoleg20/afs-ml`

Navigate to web page in browser and copy URL to use in afs app as `IMAGE_PROCESSING_URI` variable.
