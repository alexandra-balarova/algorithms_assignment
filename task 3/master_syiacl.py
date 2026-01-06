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
            ip= line
            slave_ips.append(ip)

    #params
    N = 400

    #matrix generation
    A = np.random.randint(0, 256, (N, N), dtype=DTYPE)
    B = np.random.randint(0, 256, (N, N), dtype=DTYPE)

    #divide A into equal vertical blocks for each slave
    

    #1 slave
    block_size_1 = N
    blocks_1 = []
    for i in range(1):
        start = i * block_size_1
        end = N if i == 1-1 else (i+1)*block_size_1
        A_block_1 = A[start:end, :]
        ip = slave_ips[i]
        blocks_1.append((ip, A_block_1, B, i))
        
    #3 slaves
    block_size_3 = N // 3
    blocks_3 = []
    for i in range(3):
        start = i * block_size_3
        end = N if i == 3-1 else (i+1)*block_size_3
        A_block_3 = A[start:end, :]
        ip = slave_ips[i]
        blocks_3.append((ip, A_block_3, B, i))


    #6 slaves
    block_size_6 = N // 6
    blocks_6 = []
    for i in range(6):
        start = i * block_size_6
        end = N if i == 6-1 else (i+1)*block_size_6
        A_block_6 = A[start:end, :]
        ip = slave_ips[i]
        blocks_6.append((ip, A_block_6, B, i))
        
    #9 slaves
    block_size_9 = N // 9
    blocks_9 = []
    for i in range(9):
        start = i * block_size_9
        end = N if i == 9-1 else (i+1)*block_size_9
        A_block_9 = A[start:end, :]
        ip = slave_ips[i]
        blocks_9.append((ip, A_block_9, B, i))

    print(F"[MASTER] Beginning matrix multiplication with 1 slave...{N}")
    
    start_block_1 = time.time()
    with mp.Pool(1) as pool:
        results_1 = pool.map(block_mulptiplication, blocks_1)
        
    total_block_1 = time.time() - start_block_1
    

    print("[MASTER] Beginning matrix multiplication with 3 slaves...")
        
    start_block_3 = time.time()
    with mp.Pool(3) as pool:
        results_3 = pool.map(block_mulptiplication, blocks_3)
        
    total_block_3 = time.time() - start_block_3
    

    print("[MASTER] Beginning matrix multiplication with 6 slaves...")
        
    start_block_6 = time.time()
    with mp.Pool(6) as pool:
        results_6 = pool.map(block_mulptiplication, blocks_6)
        
    total_block_6 = time.time() - start_block_6
    

    print("[MASTER] Beginning matrix multiplication with 9 slaves...")
    
    start_block_9 = time.time()
    with mp.Pool(9) as pool:
        results_9 = pool.map(block_mulptiplication, blocks_9)
        
    total_block_9 = time.time() - start_block_9
    
    
    
    filename = f"results_syiacl400.txt"
    with open(filename, "w") as f:
        f.write(f"Matrix size: {N} x {N}\n")
        f.write(f"Block MM with {1} slaves\n")
        f.write(f"Time taken: {total_block_1:.3f} seconds\n")
        f.write(f"Block MM with {3} slaves\n")
        f.write(f"Time taken: {total_block_3:.3f} seconds\n")
        f.write(f"Block MM with {6} slaves\n")
        f.write(f"Time taken: {total_block_6:.3f} seconds\n")
        f.write(f"Block MM with {9} slaves\n")
        f.write(f"Time taken: {total_block_9:.3f} seconds\n")



    print(f"[MASTER] Result saved to {filename}")
if __name__ == "__main__":
    main()
