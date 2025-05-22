# SAM2 Browser Image Segmentation

## Overview

This is a minimal web application for interactive image segmentation using the Segment Anything Model 2 (SAM2) running entirely in the browser using ONNX Runtime Web.

## Features

- Upload images via file input
- Interactive point-based image segmentation
- Toggle between positive and negative points
- Export segmentation mask as PNG

## Prerequisites

- Node.js (v16 or later)
- npm

## Setup and Installation

1. Clone the repository
```bash
git clone https://github.com/nbcaliperai/SAM.git
cd sam2-browser-segmentation
```

2. Install dependencies
```bash
npm install
```

3. Start the development server
```bash
npm run dev
```

4. Open your browser and navigate to `http://localhost:5173`

## Usage

1. Upload an image using the file input
2. Click on the image to add segmentation points
   - Green points: Positive points (include in segmentation)
   - Red points: Negative points (exclude from segmentation)
3. Use the "Toggle Point Type" button to switch between point types
4. Click "Export Mask" to download the segmentation mask
