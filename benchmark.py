import torch
import time
from loguru import logger
import numpy as np
import tqdm

# # Tensor moving benchmark

# for _ in range(10):
#     tensor_1 = torch.randint(0, 255, (1920, 1080, 3), dtype=torch.uint8)
#     st = time.time()
#     cuda_tensor = tensor_1.cuda()
#     torch.cuda.synchronize()
#     logger.info(f"Torch time to cuda: {(time.time() - st)*1000:.2f} ms")

# logger.info(f"#"*30)

# for _ in range(10):
#     arr_1 = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
#     st = time.time()
#     cuda_tensor = torch.from_numpy(arr_1).cuda()
#     torch.cuda.synchronize()
#     logger.info(f"Numpy time to cuda: {(time.time() - st)*1000:.2f} ms")

# Tensor computing benchmark

size = 100
runs = 100
foo_tenosr = torch.rand((size, size)).cuda()

for _ in tqdm.tqdm(range(runs)):
    tensor_1 = torch.rand((size, size)).cuda()
    tensor_2 = torch.rand((size, size)).cuda()

    st = time.time()
    tensor_1.matmul(tensor_2)
    torch.cuda.synchronize()
    # logger.info(f"Torch cuda computing time: {(time.time() - st)*1000:.2f} ms")

logger.info(f"#"*30)

for _ in tqdm.tqdm(range(runs)):
    tensor_1 = torch.rand((size, size))
    tensor_2 = torch.rand((size, size))

    st = time.time()
    tensor_1.matmul(tensor_2)
    torch.cuda.synchronize()
    # logger.info(f"Torch cpu computing time: {(time.time() - st)*1000:.2f} ms")

logger.info(f"#"*30)

for _ in tqdm.tqdm(range(runs)):
    arr_1 = np.random.rand(size, size)
    arr_2 = np.random.rand(size, size)

    st = time.time()
    np.matmul(arr_1, arr_2)
    # logger.info(f"Numpy computing time: {(time.time() - st)*1000:.2f} ms")

