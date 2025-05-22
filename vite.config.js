import { defineConfig } from 'vite';

export default defineConfig({
    // Configure static asset handling for ONNX models
    assetsInclude: ['**/*.ort', '**/*.onnx'],
    
    // Ensure correct handling of WebAssembly modules
    optimizeDeps: {
        exclude: ['onnxruntime-web']
    }
}); 