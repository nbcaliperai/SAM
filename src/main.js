// @ts-nocheck
import * as ort from 'onnxruntime-web';

const ENCODER_MODEL_URL = 'https://storage.googleapis.com/lb-artifacts-testing-public/sam2/sam2_hiera_tiny.encoder.ort';
const DECODER_MODEL_URL = 'https://storage.googleapis.com/lb-artifacts-testing-public/sam2/sam2_hiera_tiny.decoder.onnx';

class MaskProcessor {
    static getMaskBoundingBox(maskData, width, height) {
        let minX = width, maxX = 0;
        let minY = height, maxY = 0;
        let hasPositivePixels = false;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const index = y * width + x;
                if (maskData[index] > 0) {
                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                    hasPositivePixels = true;
                }
            }
        }

        return hasPositivePixels ? [minX, minY, maxX, maxY] : null;
    }

    static resizeMask(maskData, srcWidth, srcHeight, destWidth, destHeight) {
        const resizedMask = new Float32Array(destWidth * destHeight);
        const scaleX = srcWidth / destWidth;
        const scaleY = srcHeight / destHeight;

        for (let y = 0; y < destHeight; y++) {
            for (let x = 0; x < destWidth; x++) {
                const srcX = Math.min(Math.floor(x * scaleX), srcWidth - 1);
                const srcY = Math.min(Math.floor(y * scaleY), srcHeight - 1);
                const srcIndex = srcY * srcWidth + srcX;
                const destIndex = y * destWidth + x;
                
                resizedMask[destIndex] = maskData[srcIndex];
            }
        }

        return resizedMask;
    }
}

class SAM2Segmenter {
    constructor() {
        this.encoder = null;
        this.predictor = null;
        this.embedding = null;
        this.isInitialized = false;
    }

    async initialize() {
        try {
            console.log('Loading encoder from:', ENCODER_MODEL_URL);
            this.encoder = await ort.InferenceSession.create(ENCODER_MODEL_URL);
            console.log('Encoder loaded successfully');
            console.log('Loading decoder from:', DECODER_MODEL_URL);
            this.predictor = await ort.InferenceSession.create(DECODER_MODEL_URL);
            console.log('Decoder loaded successfully');
            this.isInitialized = true;
            return true;
        } catch (error) {
            console.error('Error initializing models:', error);
            console.error('Error details:', error.message, error.code);
            throw error;
        }
    }

    preprocessImage(image) {
        // Create a canvas to resize image to 1024x1024
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 1024;
        canvas.height = 1024;

        // Draw image scaled to fit 1024x1024
        ctx.drawImage(image, 0, 0, 1024, 1024);
        
        // Get image data
        const imageData = ctx.getImageData(0, 0, 1024, 1024);
        const { data } = imageData;

        // Convert to tensor format [1, 3, 1024, 1024]
        const float32Data = new Float32Array(3 * 1024 * 1024);
        
        // Normalize and arrange in CHW format
        for (let i = 0; i < 1024 * 1024; i++) {
            // Normalize to [-1, 1] range as expected by SAM2
            float32Data[i] = (data[i * 4] / 255.0) * 2.0 - 1.0;                    // R
            float32Data[i + 1024 * 1024] = (data[i * 4 + 1] / 255.0) * 2.0 - 1.0; // G
            float32Data[i + 2 * 1024 * 1024] = (data[i * 4 + 2] / 255.0) * 2.0 - 1.0; // B
        }

        return new ort.Tensor('float32', float32Data, [1, 3, 1024, 1024]);
    }

    async encode(image) {
        if (!this.isInitialized) {
            throw new Error('SAM2Segmenter not initialized');
        }

        const imageTensor = this.preprocessImage(image);
        const feeds = { image: imageTensor };
        
        console.log('Running encoder...');
        const results = await this.encoder.run(feeds);
        this.embedding = results.image_embed;
        console.log('Encoding complete');
        
        return this.embedding;
    }

    normalizeCoordinates(x, y, imageWidth, imageHeight) {
        // Scale coordinates to the 1024x1024 model input space
        const scaledX = (x / imageWidth) * 1024;
        const scaledY = (y / imageHeight) * 1024;
        return [scaledX, scaledY];
    }

    async predict(embedding, points, labels) {
        if (!this.isInitialized) {
            throw new Error('SAM2Segmenter not initialized');
        }

        // Prepare point coordinates and labels
        const pointCoords = new ort.Tensor('float32', 
            new Float32Array(points.flat()), 
            [1, points.length, 2]
        );
        
        const pointLabels = new ort.Tensor('float32', 
            new Float32Array(labels), 
            [1, labels.length]
        );

        // Prepare other required inputs with correct dimensions
        const maskInput = new ort.Tensor('float32', 
            new Float32Array(1 * 1 * 256 * 256).fill(0), 
            [1, 1, 256, 256]
        );
        
        const hasMaskInput = new ort.Tensor('float32', 
            new Float32Array([0.0]), 
            [1]
        );

        // High resolution features (these might need adjustment based on actual model)
        const highResFeats0 = new ort.Tensor('float32', 
            new Float32Array(1 * 32 * 256 * 256).fill(0), 
            [1, 32, 256, 256]
        );
        
        const highResFeats1 = new ort.Tensor('float32', 
            new Float32Array(1 * 64 * 128 * 128).fill(0), 
            [1, 64, 128, 128]
        );

        const feeds = {
            image_embed: embedding,
            point_coords: pointCoords,
            point_labels: pointLabels,
            mask_input: maskInput,
            has_mask_input: hasMaskInput,
            high_res_feats_0: highResFeats0,
            high_res_feats_1: highResFeats1
        };

        console.log('Running prediction...');
        const results = await this.predictor.run(feeds);
        console.log('Prediction complete');
        
        return results;
    }
}

class ImageSegmentationApp {
    constructor() {
        this.segmenter = new SAM2Segmenter();
        this.sourceCanvas = document.getElementById('sourceCanvas');
        this.imageInput = document.getElementById('imageInput');
        this.exportMaskBtn = document.getElementById('exportMask');
        this.statusDiv = document.getElementById('status');

        this.sourceCtx = this.sourceCanvas.getContext('2d');

        this.image = null;
        this.embedding = null;
        this.points = [];
        this.labels = [];
        this.currentMask = null;

        this.initializeEventListeners();
    }

    async initialize() {
        try {
            await this.segmenter.initialize();
            this.updateStatus('Models loaded successfully! Upload an image to start.', 'ready');
        } catch (error) {
            this.updateStatus('Error loading models: ' + error.message, 'error');
            console.error('Initialization error:', error);
        }
    }

    updateStatus(message, type = 'loading') {
        this.statusDiv.textContent = message;
        this.statusDiv.className = `status ${type}`;
    }

    initializeEventListeners() {
        this.imageInput.addEventListener('change', this.handleImageUpload.bind(this));
        this.sourceCanvas.addEventListener('click', this.handleCanvasClick.bind(this));
        this.sourceCanvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.handleCanvasClick(e, true);
        });
        this.exportMaskBtn.addEventListener('click', this.exportMask.bind(this));
    }

    async handleImageUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        this.updateStatus('Loading image...', 'loading');

        const reader = new FileReader();
        reader.onload = async (event) => {
            this.image = new Image();
            this.image.onload = async () => {
                await this.processImage();
            };
            this.image.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }

    async processImage() {
        try {
            // Reset state
            this.points = [];
            this.labels = [];
            this.currentMask = null;

            // Setup canvas
            const maxWidth = 800;
            const maxHeight = 600;
            const scale = Math.min(maxWidth / this.image.width, maxHeight / this.image.height, 1);
            
            this.sourceCanvas.width = this.image.width * scale;
            this.sourceCanvas.height = this.image.height * scale;

            // Draw image
            this.sourceCtx.clearRect(0, 0, this.sourceCanvas.width, this.sourceCanvas.height);
            this.sourceCtx.drawImage(this.image, 0, 0, this.sourceCanvas.width, this.sourceCanvas.height);

            this.updateStatus('Encoding image...', 'loading');

            // Encode image
            this.embedding = await this.segmenter.encode(this.image);

            this.updateStatus('Ready! Click on the image to segment. Right-click for negative points.', 'ready');
            this.exportMaskBtn.disabled = false;

        } catch (error) {
            this.updateStatus('Error processing image: ' + error.message, 'error');
            console.error('Image processing error:', error);
        }
    }

    // Helper method to check if a click is near an existing point
    findPointAtPosition(x, y, tolerance = 15) {
        for (let i = 0; i < this.points.length; i++) {
            // Convert normalized coordinates back to canvas coordinates
            const canvasX = (this.points[i][0] / 1024) * this.sourceCanvas.width;
            const canvasY = (this.points[i][1] / 1024) * this.sourceCanvas.height;
            
            const distance = Math.sqrt(Math.pow(x - canvasX, 2) + Math.pow(y - canvasY, 2));
            if (distance <= tolerance) {
                return i;
            }
        }
        return -1;
    }

    async handleCanvasClick(e, isNegative = false) {
        if (!this.image || !this.embedding) return;

        const rect = this.sourceCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Check if we're clicking on an existing point
        const existingPointIndex = this.findPointAtPosition(x, y);
        
        if (existingPointIndex !== -1) {
            // Remove the existing point
            this.points.splice(existingPointIndex, 1);
            this.labels.splice(existingPointIndex, 1);
            
            this.updateStatus('Point removed. Processing segmentation...', 'loading');
            
            // If we have remaining points, re-run prediction
            if (this.points.length > 0) {
                try {
                    const results = await this.segmenter.predict(this.embedding, this.points, this.labels);
                    const maskTensor = results.masks;
                    const maskData = maskTensor.data;
                    
                    const resizedMask = MaskProcessor.resizeMask(
                        maskData,
                        256, 256,
                        this.sourceCanvas.width,
                        this.sourceCanvas.height
                    );

                    this.currentMask = resizedMask;
                    this.redrawCanvas();
                    this.updateStatus(`Point removed. Remaining points: ${this.points.length}`, 'ready');
                } catch (error) {
                    this.updateStatus('Segmentation error: ' + error.message, 'error');
                    console.error('Segmentation error:', error);
                }
            } else {
                // No points left, clear mask
                this.currentMask = null;
                this.redrawCanvas();
                this.updateStatus('All points removed. Click to add new points.', 'ready');
            }
            return;
        }

        // Convert canvas coordinates to original image coordinates
        const scaleX = this.image.width / this.sourceCanvas.width;
        const scaleY = this.image.height / this.sourceCanvas.height;
        const originalX = x * scaleX;
        const originalY = y * scaleY;

        // Normalize coordinates for the model
        const [normalizedX, normalizedY] = this.segmenter.normalizeCoordinates(
            originalX, originalY, this.image.width, this.image.height
        );

        // Add point and label
        this.points.push([normalizedX, normalizedY]);
        this.labels.push(isNegative ? 0 : 1);

        this.updateStatus('Processing segmentation...', 'loading');

        try {
            // Run prediction with all accumulated points
            const results = await this.segmenter.predict(this.embedding, this.points, this.labels);

            // Get the mask (usually the first mask is the best one)
            const maskTensor = results.masks;
            const maskData = maskTensor.data;
            
            // The mask is typically 256x256, resize to canvas size
            const resizedMask = MaskProcessor.resizeMask(
                maskData,
                256, 256,  // SAM2 typically outputs 256x256 masks
                this.sourceCanvas.width,
                this.sourceCanvas.height
            );

            this.currentMask = resizedMask;

            // Redraw everything
            this.redrawCanvas();

            this.updateStatus(`Segmentation complete! Points: ${this.points.length} (Click on existing points to remove them)`, 'ready');

        } catch (error) {
            this.updateStatus('Segmentation error: ' + error.message, 'error');
            console.error('Segmentation error:', error);
        }
    }

    redrawCanvas() {
        // Clear and redraw original image
        this.sourceCtx.clearRect(0, 0, this.sourceCanvas.width, this.sourceCanvas.height);
        this.sourceCtx.drawImage(this.image, 0, 0, this.sourceCanvas.width, this.sourceCanvas.height);

        // Draw mask overlay if available
        if (this.currentMask) {
            this.drawMaskOverlay(this.currentMask);
        }

        // Draw all points
        this.drawAllPoints();

        // Draw bounding box
        if (this.currentMask) {
            const bbox = MaskProcessor.getMaskBoundingBox(
                this.currentMask,
                this.sourceCanvas.width,
                this.sourceCanvas.height
            );
            if (bbox) {
                this.drawBoundingBox(bbox);
            }
        }
    }

    drawAllPoints() {
        this.points.forEach((point, index) => {
            const label = this.labels[index];
            const isNegative = label === 0;
            
            // Convert normalized coordinates back to canvas coordinates
            const canvasX = (point[0] / 1024) * this.sourceCanvas.width;
            const canvasY = (point[1] / 1024) * this.sourceCanvas.height;
            
            this.drawPoint(canvasX, canvasY, isNegative);
        });
    }

    drawPoint(x, y, isNegative = false) {
        this.sourceCtx.fillStyle = isNegative ? '#ff4444' : '#44ff44';
        this.sourceCtx.strokeStyle = 'white';
        this.sourceCtx.lineWidth = 2;
        
        this.sourceCtx.beginPath();
        this.sourceCtx.arc(x, y, 6, 0, 2 * Math.PI);
        this.sourceCtx.fill();
        this.sourceCtx.stroke();
        
        // Add a small border to make points more visible and clickable
        this.sourceCtx.strokeStyle = '#000000';
        this.sourceCtx.lineWidth = 1;
        this.sourceCtx.stroke();
    }

    drawMaskOverlay(maskData) {
        const imageData = this.sourceCtx.getImageData(0, 0, this.sourceCanvas.width, this.sourceCanvas.height);
        const data = imageData.data;

        // Apply mask overlay with semi-transparent color
        for (let i = 0; i < this.sourceCanvas.width * this.sourceCanvas.height; i++) {
            if (maskData[i] > 0.5) {
                const pixelIndex = i * 4;
                // Apply blue-ish overlay
                data[pixelIndex] = Math.floor(data[pixelIndex] * 0.7 + 100 * 0.3);     // R
                data[pixelIndex + 1] = Math.floor(data[pixelIndex + 1] * 0.7 + 150 * 0.3); // G
                data[pixelIndex + 2] = Math.floor(data[pixelIndex + 2] * 0.7 + 255 * 0.3); // B
            }
        }

        this.sourceCtx.putImageData(imageData, 0, 0);
    }

    drawBoundingBox(bbox) {
        const [x1, y1, x2, y2] = bbox;
        
        this.sourceCtx.strokeStyle = '#00ff00';
        this.sourceCtx.lineWidth = 3;
        this.sourceCtx.setLineDash([5, 5]);
        this.sourceCtx.beginPath();
        this.sourceCtx.rect(x1, y1, x2 - x1, y2 - y1);
        this.sourceCtx.stroke();
        this.sourceCtx.setLineDash([]);
    }

    exportMask() {
        if (!this.sourceCanvas) return;

        const link = document.createElement('a');
        link.download = 'sam2_segmentation.png';
        link.href = this.sourceCanvas.toDataURL('image/png');
        link.click();
    }
}

// Initialize the application
async function main() {
    const app = new ImageSegmentationApp();
    await app.initialize();
}

// Start the application when the page loads
window.addEventListener('load', main);