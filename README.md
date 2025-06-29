## QAE Architectures in PennyLane+PyTorch

### Aims
*Investigation of quantum autoencoder architectures for effective denoising of signal and time series.*

### Current work
*Explore this repository and its files*
- Investigate ways to implement encoder and decoder in PennyLane.
- As we all may end up having slightly different projects (to be discussed),
  at this stage keep your draft work in your separate folders.

### Folders
*We'll need some common utilities, I suggest to keep them as .py files in a directory.*
- jacobs_examples: examples of QAE and supporting code from Jacob
- logs: this folder may be created to hold saved data, training history, plots, etc.
- qae_utils: which is a collection of Python utilities to include
  - Charts.py - functions plotting time-series data (fancy and flexible)
  - Files.py - functions saving time-series and support data to disk
  - Tools.py - some odd collection of utilities, including extras for PennyLane
  - Window.py - functions creating and managing sliding windows (making, splitting, etc.)
