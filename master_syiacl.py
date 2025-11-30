import socket
import struct
from functools import total_ordering

import numpy as np
import time
import multiprocessing as mp
import os

PORT = 55055
DTYPE = np.uint8

def recv_exact(conn, nbytes):
    buf = b""
    while len(buf) < nbytes:
        chunk = conn.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("Socket disconnected.")
        buf += chunk
    return buf

def block_mulptiplication(args):
    ip, A_block, B, block_index = args

    rowsA, colsA = A_block.shape
    rowsB, colsB = B.shape

    s = socket.socket()
    s.connect((ip, PORT))

    # Send header
    header = struct.pack("!4i", rowsA, colsA, rowsB, colsB)
    s.sendall(header)

    # Send matrix blocks
    s.sendall(A_block.tobytes())
    s.sendall(B.tobytes())

    # Receive result block
    shape_data = recv_exact(s, 8)
    r, c = struct.unpack("!2i", shape_data)

    data_raw = recv_exact(s, r*c)
    C_block = np.frombuffer(data_raw, dtype=DTYPE).reshape(r, c)

    s.close()
    return (block_index, C_block)

def regular_multiplication(A, B, N):
    C = np.zeros((N, N), dtype=np.uint32)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += int(A[i, k]) * int(B[k, j])

    # C = (A.astype(np.uint32) @ B.astype(np.uint32))
    return C


def main():
    if not os.path.exists("master_ip.txt"):
        raise FileNotFoundError("master_ip.txt missing.")

    with open("master_ip.txt", "r") as f:
        slave_ips = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            ip, port = line.split(":")
            slave_ips.append((ip, int(port)))

    num_slaves = len(slave_ips)
    print(f"[MASTER] Found {num_slaves} slaves.")

    #params
    N = 500
    exp_info = f"{num_slaves}s"

    #matrix generation
    A = np.random.randint(0, 256, (N, N), dtype=DTYPE)
    B = np.random.randint(0, 256, (N, N), dtype=DTYPE)

    #divide A into equal vertical blocks for each slave
    block_size = N // num_slaves
    blocks = []

    for i in range(num_slaves):
        start = i * block_size
        end = N if i == num_slaves-1 else (i+1)*block_size
        A_block = A[start:end, :]
        ip, port = slave_ips[i]
        blocks.append((ip, port, A_block, B, i))


    print("[MASTER] Beginning regular matrix multiplication...")

    start_regular = time.time()

    C_regular = regular_multiplication(A, B, N)

    total_regular = time.time() - start_regular


    print("[MASTER] Beginning block matrix multiplication...")

    start_block = time.time()

    # --- parallel execution ---
    with mp.Pool(num_slaves) as pool:
        results = pool.map(block_mulptiplication, blocks)

    #sort results by block index
    results.sort(key=lambda x: x[0])
    #reconstruct full matrix C
    C_block = np.vstack([block for _, block in results])

    total_block = time.time() - start_block

    print(f"[MASTER] Completed in {total_block:.3f} seconds.")


    filename = f"results_syiacl_{exp_info}.txt"
    with open(filename, "w") as f:
        f.write(f"Matrix size: {N} x {N}\n")
        f.write(f"Block MM with {num_slaves} slaves\n")
        f.write(f"Time taken: {total_block:.3f} seconds\n")
        f.write(f"Regular MM\n")
        f.write(f"Time taken: {total_regular:.3f} seconds\n")

    print(f"[MASTER] Result saved to {filename}")
if __name__ == "__main__":
    main()
