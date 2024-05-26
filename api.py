import os
import logging
import multiprocessing as mp

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, Request
import torch
from ultralytics import YOLO

from utils import get_rooms

logging.basicConfig(filename='afs-ml.log', level=logging.INFO)
logger = logging.getLogger(__file__)


app = FastAPI(debug=False)


MODEL = YOLO('runs/segment/train/weights/best.pt')


@app.on_event("startup")
async def startup_event():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # MODEL = YOLO('runs/segment/train/weights/best.pt')
    logger.info(f'model loaded: {MODEL.info()}')

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
    logger.addHandler(handler)

    try:
        stream = os.popen('nvcc --version')
        logger.info(f'nvcc --version:\n {stream.read()}')
    except:
        logger.info('Command nvcc not found')

    try:
        stream = os.popen('nvidia-smi')
        logger.info(f'nvidia-smi:\n {stream.read()}')
    except:
        logger.info('Command nvidia-smi not found')


@app.get("/")
async def find_room_contours():
    return {
        "Message": "Ready for inference."
    }


@app.post("/")
async def find_room_contours(request: Request):
    file = await request.body()
    logger.info(f'Body size: {len(file)}')

    logger.info(f'Number of available CPUs: {mp.cpu_count()}')

    # set device to cuda if available
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f'Cuda device is available: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 10**9:.3f} GB).')
    else:
        logger.info(f'No cuda devices were found.')

    # load image
    nparr = np.frombuffer(file, np.uint8)
    logger.info(f'np array shape: {nparr.shape}')
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # image height/width
    h, w = img.shape[0], img.shape[1]

    # process image
    rooms = get_rooms(img, MODEL, area_threshold=0.001)

    # np.ndarrays into lists
    rooms = [room.squeeze(axis=-2).tolist() for room in rooms]

    # absolute coordinates to relative
    rel_rooms = [
        [
            [x / w, y / h]
            for x, y in room
        ]
        for room in rooms
    ]

    return {
        "rooms": rel_rooms
    }


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_grad_enabled(False)
    uvicorn.run(app, host='0.0.0.0', port=80)
