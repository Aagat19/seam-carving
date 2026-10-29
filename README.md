# Seam Carving: Content-Aware Image Resizing

A C++ implementation of seam carving (Avidan & Shamir, 2007). Instead of scaling or cropping, the image is shrunk by repeatedly removing the lowest-energy *seam*, a connected path of pixels running from top to bottom (or left to right). Low-energy regions such as sky or flat backgrounds are removed first, while edges and objects are preserved.

The energy computation and the dynamic program are written from scratch. OpenCV is used only for loading, saving and displaying images and for the `Mat` container.

## How it works

1. **Energy.** Each pixel gets a dual-gradient energy: the squared RGB differences between its left/right neighbours plus its up/down neighbours, square-rooted. Border pixels use clamped neighbours.
2. **Dynamic programming.** For a vertical seam, `M[i][j] = energy[i][j] + min(M[i-1][j-1], M[i-1][j], M[i-1][j+1])`. A parent matrix `P` stores which of the three cells was chosen.
3. **Backtracking.** Start from the minimum value in the last row of `M` and follow `P` upward to recover the seam.
4. **Removal.** Shift each row left over the seam pixel and drop the last column.
5. **Incremental energy update.** Only pixels next to the removed seam get new neighbours, so only their energy is recomputed, not the whole image.
6. **Horizontal seams** reuse the vertical routine by transposing the image, removing a vertical seam, and transposing back.

**Complexity:** O(H × W) per seam, so O(k × H × W) to remove k seams.

## Build

Requires a C++17 compiler and OpenCV 4.

```bash
g++ -std=c++17 -O2 seam_carving.cpp -o seam_carving $(pkg-config --cflags --libs opencv4)
```

## Run

```bash
./seam_carving
```

The program asks for the image path and the target height and width (each must be at most the original size). It shows progress, saves the result to `output.jpg`, and opens it in a window.

## Limitations

- Only shrinking is supported (no seam insertion for enlarging).
- All vertical seams are removed before the horizontal ones, rather than choosing the optimal order.
- The incremental energy update recomputes only the same row's neighbours of each removed pixel, which is a close approximation of a full recompute.
