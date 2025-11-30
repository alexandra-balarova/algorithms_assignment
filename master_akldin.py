#!/usr/bin/env python3
import socket
import struct
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

def worker_process(args):
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


def main():
    # Load list of slave IPs
    if not os.path.exists("master_ip.txt"):
        raise FileNotFoundError("master_ip.txt missing.")

    with open("master_ip.txt", "r") as f:
        slave_ips = [line.strip() for line in f if line.strip()]

    num_slaves = len(slave_ips)
    print(f"[MASTER] Found {num_slaves} slaves.")

    # ----- Parameters -----
    N = 2000   # Use smaller during testing; enlarge on experiment day
    exp_info = f"{num_slaves}s"

    # Generate matrices
    A = np.random.randint(0, 256, (N, N), dtype=DTYPE)
    B = np.random.randint(0, 256, (N, N), dtype=DTYPE)

    # Divide A into equal vertical blocks for each slave
    block_size = N // num_slaves
    blocks = []

    for i in range(num_slaves):
        start = i * block_size
        end = N if i == num_slaves-1 else (i+1)*block_size
        A_block = A[start:end, :]
        blocks.append((slave_ips[i], A_block, B, i))

    print("[MASTER] Beginning distributed matrix multiplication...")

    start_time = time.time()

    # --- Parallel execution ---
    with mp.Pool(num_slaves) as pool:
        results = pool.map(worker_process, blocks)

    # Sort results by block index
    results.sort(key=lambda x: x[0])

    # Reconstruct full matrix C
    C = np.vstack([blk for _, blk in results])

    total_time = time.time() - start_time

    print(f"[MASTER] Completed in {total_time:.3f} seconds.")

    # Save results to file
    filename = f"results_akldin_{exp_info}.txt"
    with open(filename, "w") as f:
        f.write(f"Distributed MM with {num_slaves} slaves\n")
        f.write(f"Matrix size: {N} x {N}\n")
        f.write(f"Time taken: {total_time:.3f} seconds\n")

    print(f"[MASTER] Result saved to {filename}")

if __name__ == "__main__":
    main()
