# Shape Detector Solution

This repository contains my implementation of the shape detection algorithm for the given assignment. The code is contained in `main.ts` and should be placed in the `src/` folder of the original project.

## How to Use

1. Clone or download the original shape-detector repository.
2. Replace the existing `src/main.ts` with the file from this repo.
3. Run `npm install` to install dependencies.
4. Run `npm run dev` to start the development server.
5. Open the provided local URL in your browser and test with the supplied images.

## Approach

The algorithm follows a classic computer vision pipeline:

- Convert the image to binary using a luminance threshold (128) to isolate dark shapes.
- Extract connected components (4‑connectivity) to obtain individual blobs.
- For each blob, compute geometric features: bounding box, convex hull, area, perimeter, circularity, concavity, corner count (via RDP simplification and angle filtering), and aspect ratio.
- Instead of a hard decision tree, a probabilistic scoring system assigns a likelihood to each shape type (circle, triangle, rectangle, pentagon, star) based on how well the features match ideal values.
- The shape with the highest score wins, and its confidence is scaled relative to the maximum possible score for that type.
- Overlapping detections are removed using non‑maximum suppression (IoU > 0.4).

The probabilistic approach proved more robust to edge cases and feature noise, at the cost of a slight increase in processing time (still well under 2000ms per image).

