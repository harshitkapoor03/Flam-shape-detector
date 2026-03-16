import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * MAIN ALGORITHM TO IMPLEMENT
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   *
   * TODO: Implement shape detection algorithm here
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();
    const { width, height, data } = imageData;

    // Started with edge detection but switched to thresholding , works better for filled shapes
    // Had to experiment with threshold value, 128 seems to work well for test images
    const binary = this.threshold(data, width, height, 128);

    // Step 2: Find connected components (blobs)
    // Tried 8 connectivity first but it merged shapes that were too close
    // Switched to 4 connectivity and that fixed it
    const blobs = this.connectedComponents(binary, width, height);

    const shapes: DetectedShape[] = [];
    const imageArea = width * height;

    // Step 3: Process each blob
    for (const blob of blobs) {
      // Filter out noise - small blobs are usually artifacts
      // Tuned this threshold by trial and error
      if (blob.length < 20) continue;

      const bbox = this.getBoundingBox(blob);
      const bboxArea = bbox.width * bbox.height;

      // Skip if blob covers most of image (background)
      if (bboxArea > imageArea * 0.8) continue;
      
      // Skip tiny blobs
      if (bboxArea < 60) continue;
      
      // Skip very thin lines (might be edges)
      const longSide = Math.max(bbox.width, bbox.height);
      const shortSide = Math.min(bbox.width, bbox.height);
      if (longSide / shortSide > 8) continue;

      // Get convex hull for shape analysis
      // Added this after realizing bounding box alone isn't enough
      const hull = this.convexHull(blob);
      const hullArea = this.shoelace(hull);
      if (hullArea < 80) continue;

      // Use centroid for better center positioning
      // Bounding box center wasn't accurate for stars and rotated shapes
      const center = this.computeCentroid(blob);
      
      // Calculate filled area (pixel count)
      const filledArea = blob.length;

      // Calculate features for classification
      // Added more features as I encountered edge cases
      const perimeter = this.perimeterOf(hull);
      const circularity = perimeter > 0 ? (4 * Math.PI * hullArea) / (perimeter * perimeter) : 0;
      const concavity = hullArea > 0 ? filledArea / hullArea : 1;
      
      const diag = Math.sqrt(bbox.width ** 2 + bbox.height ** 2);
      const epsilon = Math.max(2, diag * 0.02);
      const simplified = this.rdpSimplify(hull, epsilon);
      const corners = this.countMeaningfulCorners(simplified);
      const aspectRatio = Math.min(bbox.width, bbox.height) / Math.max(bbox.width, bbox.height);

      // Originally used a simple decision tree, but wanted more nuance
      // Switched to probability-based approach to handle ambiguous cases better
      const scores = {
        circle: this.calculateCircleProb(circularity),
        triangle: this.calculateTriangleProb(corners, circularity),
        rectangle: this.calculateRectangleProb(corners, aspectRatio, hullArea, bbox),
        pentagon: this.calculatePentagonProb(corners, concavity, circularity),
        star: this.calculateStarProb(concavity, corners, circularity)
      };

      // Find the shape type with maximum probability
      const { type, confidence } = this.getMaxProbability(scores);
            shapes.push({
        type, 
        confidence, 
        boundingBox: bbox, 
        center, 
        area: filledArea
      });
    }

    const processingTime = performance.now() - startTime;

    // Remove overlapping detections
    // Added this after noticing duplicate detections on some images
    const finalShapes = this.deduplicateShapes(shapes);

    return {
      shapes: finalShapes,
      processingTime,
      imageWidth: width,
      imageHeight: height,
    };
  }

  // Probability calculation functions
  // Tuned these by running the evaluator repeatedly and adjusting
  private calculateCircleProb(circularity: number): number {
    // Circle probability peaks at circularity = 1.0
    // Gaussian-like function - wanted a smooth drop-off
    return Math.exp(-Math.pow((circularity - 1.0) * 10, 2)) * 0.95;
  }

  private calculateTriangleProb(corners: number, circularity: number): number {
    if (corners !== 3) return 0.1; // Base probability if corner count doesn't match
    // Triangles have lower circularity than circles but higher than stars
    const circularityScore = Math.max(0, 1 - Math.abs(circularity - 0.6) * 1.5);
    return 0.85 * circularityScore;
  }

  private calculateRectangleProb(corners: number, aspectRatio: number, hullArea: number, bbox: any): number {
    if (corners !== 4) return 0.1;
    
    // Rectangles should fill their bounding box well
    const bboxFill = hullArea / (bbox.width * bbox.height);
    
    // Aspect ratio close to 1 for squares, but rectangles can have any ratio
    // Using a broad tolerance so we don't miss rectangles
    const aspectScore = Math.min(1, aspectRatio * 1.5);
    
    return 0.8 * bboxFill * (0.5 + 0.5 * aspectScore);
  }

  private calculatePentagonProb(corners: number, concavity: number, circularity: number): number {
    if (corners !== 5 && corners !== 6) return 0.1;
    
    // Pentagons are convex (concavity close to 1) and have moderate circularity
    const convexScore = Math.min(1, concavity);
    const circularityScore = Math.max(0, 1 - Math.abs(circularity - 0.85) * 3);
    
    return 0.8 * convexScore * circularityScore;
  }

  private calculateStarProb(concavity: number, corners: number, circularity: number): number {
    // Stars are concave (concavity < 0.8) and have moderate corners (5-10)
    // This was the trickiest to tune - had to analyze pixel data
    const concavityScore = Math.max(0, Math.min(1, (0.8 - concavity) * 3));
    const cornerScore = corners >= 5 && corners <= 10 ? 1 : 0.3;
    const circularityScore = Math.max(0, 1 - Math.abs(circularity - 0.7) * 2);
    
    return 0.9 * concavityScore * cornerScore * (0.7 + 0.3 * circularityScore);
  }

  // Find shape with maximum probability
  private getMaxProbability(scores: {
  circle: number;
  triangle: number;
  rectangle: number;
  pentagon: number;
  star: number;
}): { type: DetectedShape["type"]; confidence: number } {
  
  // 1. Find which shape has the highest raw score
  const entries = Object.entries(scores) as [DetectedShape["type"], number][];
  let maxType: DetectedShape["type"] = "circle";
  let maxScore = 0;
  for (const [type, score] of entries) {
    if (score > maxScore) {
      maxScore = score;
      maxType = type;
    }
  }

  // 2. Define the maximum possible raw score for each shape
  const maxPossible: Record<DetectedShape["type"], number> = {
    circle: 0.95,
    triangle: 0.85,
    rectangle: 0.8,
    pentagon: 0.8,
    star: 0.9
  };

  // 3. Scale the confidence: raw score / max possible for that shape
  let confidence = maxScore / maxPossible[maxType];

  // 4. Clamp to a reasonable range (e.g., 0.2 – 0.99) to avoid extreme lows/highs
  confidence = Math.min(0.99, Math.max(0.2, confidence));

  return { type: maxType, confidence };
}

  // Helper method computes center of mass
  // Added after reading about centroid vs bounding box center
  private computeCentroid(blob: Point[]): Point {
    let sumX = 0, sumY = 0;
    for (const p of blob) {
      sumX += p.x;
      sumY += p.y;
    }
    return {
      x: sumX / blob.length,
      y: sumY / blob.length,
    };
  }

  // Basic thresholding  kept simple but effective
  // Started with Otsu's method but it was overkill for these images
  private threshold(data: Uint8ClampedArray, w: number, h: number, thresh: number): Uint8Array {
    const bin = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) {
      const r = data[i*4], g = data[i*4+1], b = data[i*4+2];
      // Standard luminance formula
      const gray = 0.299*r + 0.587*g + 0.114*b;
      bin[i] = gray < thresh ? 1 : 0;
    }
    return bin;
  }


  private connectedComponents(bin: Uint8Array, w: number, h: number): Point[][] {
    const visited = new Uint8Array(w * h);
    const blobs: Point[][] = [];

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        if (!bin[idx] || visited[idx]) continue;

        const blob: Point[] = [];
        const queue = [idx];
        visited[idx] = 1;

        while (queue.length > 0) {
          const cur = queue.pop()!;
          const cx = cur % w, cy = Math.floor(cur / w);
          blob.push({ x: cx, y: cy });

          // Check 4 neighbors (4-connectivity)
          const neighbours = [
            cy > 0 ? (cy-1)*w+cx : -1,
            cy < h-1 ? (cy+1)*w+cx : -1,
            cx > 0 ? cy*w+(cx-1) : -1,
            cx < w-1 ? cy*w+(cx+1) : -1,
          ];
          
          for (const ni of neighbours) {
            if (ni >= 0 && bin[ni] && !visited[ni]) {
              visited[ni] = 1;
              queue.push(ni);
            }
          }
        }
        blobs.push(blob);
      }
    }
    return blobs;
  }

  // Simple bounding box calculation
  private getBoundingBox(pts: Point[]) {
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    for (const p of pts) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }

    return {
      x: minX,
      y: minY,
      width: maxX - minX + 1,
      height: maxY - minY + 1,
    };
  }

  // Shoelace formula for polygon area

  private shoelace(pts: Point[]): number {
    let area = 0;
    const n = pts.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
    }
    return Math.abs(area) / 2;
  }

  // Andrew's monotone chain convex hull
  // Found this algorithm online  
  private convexHull(points: Point[]): Point[] {
    const pts = [...points].sort((a,b) => a.x !== b.x ? a.x-b.x : a.y-b.y);
    
    const cross = (o:Point, a:Point, b:Point) =>
      (a.x-o.x)*(b.y-o.y) - (a.y-o.y)*(b.x-o.x);
    
    const lower: Point[] = [];
    for (const p of pts) {
      while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0)
        lower.pop();
      lower.push(p);
    }
    
    const upper: Point[] = [];
    for (let i = pts.length-1; i >= 0; i--) {
      const p = pts[i];
      while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0)
        upper.pop();
      upper.push(p);
    }
    
    upper.pop(); 
    lower.pop();
    return lower.concat(upper);
  }

  // Ramer-Douglas-Peucker simplification
  // Added this to reduce noise in corner detection
  private rdpSimplify(points: Point[], epsilon: number): Point[] {
    if (points.length <= 2) return points;
    
    const start = points[0], end = points[points.length-1];
    let maxDist = 0, maxIdx = 0;
    
    for (let i = 1; i < points.length-1; i++) {
      const d = this.ptLineDist(points[i], start, end);
      if (d > maxDist) { 
        maxDist = d; 
        maxIdx = i; 
      }
    }
    
    if (maxDist > epsilon) {
      const left = this.rdpSimplify(points.slice(0, maxIdx+1), epsilon);
      const right = this.rdpSimplify(points.slice(maxIdx), epsilon);
      return left.slice(0,-1).concat(right);
    }
    return [start, end];
  }

  // Distance from point to line segment
  private ptLineDist(p: Point, a: Point, b: Point): number {
    const dx = b.x-a.x, dy = b.y-a.y;
    const lenSq = dx*dx + dy*dy;
    if (lenSq === 0) return Math.hypot(p.x-a.x, p.y-a.y);
    const t = ((p.x-a.x)*dx + (p.y-a.y)*dy) / lenSq;
    const tClamped = Math.max(0, Math.min(1, t));
    const projX = a.x + tClamped * dx;
    const projY = a.y + tClamped * dy;
    return Math.hypot(p.x - projX, p.y - projY);
  }

  // Calculate perimeter
  private perimeterOf(pts: Point[]): number {
    let p = 0;
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i+1) % pts.length];
      p += Math.hypot(b.x-a.x, b.y-a.y);
    }
    return p;
  }

  // Count corners with angle < 140 degrees
  // Had to experiment with the angle threshold - 140 worked best for our use case where we are detcting max pentagon and stars
  private countMeaningfulCorners(pts: Point[]): number {
    const n = pts.length;
    if (n <= 2) return n;
    
    let count = 0;
    for (let i = 0; i < n; i++) {
      const a = pts[(i-1+n)%n], b = pts[i], c = pts[(i+1)%n];
      const v1x = a.x-b.x, v1y = a.y-b.y;
      const v2x = c.x-b.x, v2y = c.y-b.y;
      
      const dot = v1x*v2x + v1y*v2y;
      const mag1 = Math.hypot(v1x, v1y);
      const mag2 = Math.hypot(v2x, v2y);
      
      if (mag1 === 0 || mag2 === 0) continue;
      
      const cosA = Math.max(-1, Math.min(1, dot/(mag1*mag2)));
      const angle = Math.acos(cosA) * 180 / Math.PI;
      
      if (angle < 140) count++;
    }
    return count;
  }

  // Remove overlapping detections
  private deduplicateShapes(shapes: DetectedShape[]): DetectedShape[] {
    const sorted = [...shapes].sort((a,b) => b.confidence - a.confidence);
    const kept: DetectedShape[] = [];
    
    for (const s of sorted) {
      if (!kept.some(k => this.iou(k.boundingBox, s.boundingBox) > 0.4)) {
        kept.push(s);
      }
    }
    return kept;
  }

  // Intersection over Union for overlap detection
  private iou(
    a: { x: number; y: number; width: number; height: number },
    b: { x: number; y: number; width: number; height: number }
  ): number {
    const ix = Math.max(0, Math.min(a.x+a.width, b.x+b.width) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y+a.height, b.y+b.height) - Math.max(a.y, b.y));
    const inter = ix * iy;
    const union = a.width*a.height + b.width*b.height - inter;
    return union > 0 ? inter/union : 0;
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});