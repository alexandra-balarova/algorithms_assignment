#!/usr/bin/env python3
import socket
import struct
import numpy as np
import os

PORT = 50503     # random port between 50000-60000
DTYPE = np.uint8   # 1-byte integers

def recv_exact(conn, nbytes):
    """Receive exactly nbytes from socket."""
    buf = b""
    while len(buf) < nbytes:
        chunk = conn.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed unexpectedly.")
        buf += chunk
    return buf

def handle_client(conn):
    # Receive header: rowsA, colsA, rowsB, colsB  (4 × int32)
    header = recv_exact(conn, 16)
    rowsA, colsA, rowsB, colsB = struct.unpack("!4i", header)

    # Receive matrix A
    sizeA = rowsA * colsA
    rawA = recv_exact(conn, sizeA)
    A = np.frombuffer(rawA, dtype=DTYPE).reshape(rowsA, colsA)

    # Receive matrix B
    sizeB = rowsB * colsB
    rawB = recv_exact(conn, sizeB)
    B = np.frombuffer(rawB, dtype=DTYPE).reshape(rowsB, colsB)

    # Compute block multiplication

    C = np.zeros((rowsA, colsB), dtype=np.uint32)

    for i in range(rowsA):
        for j in range(colsB):
            for k in range(colsA):

                C[i, j] += int(A[i, k]) * int(B[k, j])

    conn.sendall(struct.pack("!2i", C.shape[0], C.shape[1]))
    conn.sendall(C.tobytes())

def main():
    # Read own IP from local file
    if not os.path.exists("../slave1/slave_ip.txt"):
        raise FileNotFoundError("slave_ip.txt is missing.")

    with open("../slave1/slave_ip.txt", "r") as f:
        my_ip = f.read().strip()

    print(f"[SLAVE] Starting server on {my_ip}:{PORT}")

    s = socket.socket()
    s.bind((my_ip, PORT))
    s.listen(5)

    print("[SLAVE] Ready and waiting for master...")

    while True:
        conn, addr = s.accept()
        print(f"[SLAVE] Connected by {addr}")
        try:
            handle_client(conn)
        except Exception as e:
            print("[SLAVE] Error:", e)
        conn.close()

if __name__ == "__main__":
    main()
