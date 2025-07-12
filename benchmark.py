import numpy as np, time, faiss
xb = np.random.random((10000, 384)).astype('float32')
xq = np.random.random((10, 384)).astype('float32')
index = faiss.IndexFlatL2(384)
index.add(xb)
t0 = time.time()
D, I = index.search(xq, k=5)
print(f"FAISS L2 Exact Search time: {(time.time() - t0)*1000:.2f} ms")
