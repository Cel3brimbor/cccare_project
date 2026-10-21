# Network

A local studio for a leg-sensor network. Training runs on Apple Metal through MLX. The browser shows the network, one window of accelerometer and muscle data, and the machine it is training on.

The recordings are [HuGaDB](https://github.com/romanchereshnev/HuGaDB): accelerometers on both feet, shins, and thighs, plus a muscle sensor on each quadriceps. A window is about 2.3 seconds. The label is the activity at the middle of that window: walking, running, stairs, sitting, standing, and standing in an elevator. People 16, 17, and 18 are held out for the test set.

The trunk is a temporal residual network, sized so a training step fills the GPU. The head is a narrow ReLU layer. The inspector still multiplies one column of that head at a time: \(z = hW + b\).

## Run

```bash
chmod +x scripts/dev.sh
./scripts/dev.sh
```

Open http://127.0.0.1:5173. The first launch downloads HuGaDB (about 85 MB) into `data/raw`, cuts it into windows, and caches them. After that, Train runs on the GPU named in the toolbar. Step holds one batch. Try a window walks through held-out people.

Metal has to be available. If it is not, training stays off.

## Archived tasks

IMDB reviews and MNIST digits live in `backend/app/archive`. They are not loaded or shown. Set `ENABLE = True` in `backend/app/archive/__init__.py` and restart the bench to put them back in the dataset switch.

## Keys

On the training bench, Space starts and pauses and Period steps while the network has focus. In the clinic, Cmd+K or Ctrl+K opens a palette to jump to any patient or action, and the arrow keys move through the patient list.
