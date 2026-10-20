# Network

A local studio for a small IMDB sentiment network. Training runs on Apple Metal through MLX. The browser shows the network, the linear algebra of a single review, and the machine it is training on.

The model is a mean-pooled embedding followed by a shallow MLP. Each hidden unit on the canvas is a real unit. A line is a weight. The inspector multiplies one column at a time: \(z = hW + b\), then ReLU or a sigmoid, then binary cross-entropy, then the backward products and the Adam or SGD step.

## Run

```bash
chmod +x scripts/dev.sh
./scripts/dev.sh
```

Open http://127.0.0.1:5173. The first launch downloads the Stanford IMDB archive (about 80 MB) into `data/raw`, builds a 12,000-word vocabulary, and caches it. After that, Train runs on the GPU named in the toolbar. Step holds one batch so the dot product can be read slowly. Try a review sends a sentence and gets back 0 or 1.

Metal has to be available. If it is not, training stays off.

## Keys

On the training bench, Space starts and pauses and Period steps while the network has focus. In the clinic, Cmd+K or Ctrl+K opens a palette to jump to any patient or action, and the arrow keys move through the patient list.
