// import nodejs bindings to native tensorflow,
// not required, but will speed up things drastically (python required)
import '@tensorflow/tfjs-node';
// implements nodejs wrappers for HTMLCanvasElement, HTMLImageElement, ImageData
import * as canvas from 'canvas';
import * as faceapi from '@vladmandic/face-api';
import fs from 'fs';
import path from 'path';

// patch nodejs environment, we need to provide an implementation of
// HTMLCanvasElement and HTMLImageElement
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData } as any);

// Fallback em caso de falha ou ausência de rostos: expressão neutra
const defaultNeutralExpression = [{
  expressions: {
    neutral: 1.0,
    happy: 0.0,
    sad: 0.0,
    angry: 0.0,
    fearful: 0.0,
    disgusted: 0.0,
    surprised: 0.0
  }
}];


await faceapi.nets.ssdMobilenetv1.loadFromDisk('./weights');
await faceapi.nets.faceLandmark68Net.loadFromDisk('./weights');
await faceapi.nets.faceExpressionNet.loadFromDisk('./weights');
const minConfidence = 0.5;
const faceDetectionOptions = new faceapi.SsdMobilenetv1Options({ minConfidence });

const framesDir = process.env.FRAMES_DIR || '../data/interim/frames/';
const processImage = async (filePath, outputJsonPath, outputImagePath) => {
  try {
    const img = await canvas.loadImage(filePath);
    const width = (img as any).width ?? (img as any).naturalWidth;
    const height = (img as any).height ?? (img as any).naturalHeight;
    if (!img || !Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
      console.warn(`Invalid image dimensions in ${filePath}, using default neutral expression`);
      fs.writeFileSync(outputJsonPath, JSON.stringify(defaultNeutralExpression, null, 2));
      return;
    }

    const imageCanvas = canvas.createCanvas(width, height);
    const ctx = imageCanvas.getContext('2d');
    if (!ctx) {
      console.warn(`Failed to get 2d context for ${filePath}, using default neutral expression`);
      fs.writeFileSync(outputJsonPath, JSON.stringify(defaultNeutralExpression, null, 2));
      return;
    }
    ctx.drawImage(img as any, 0, 0, width, height);

    const results = await faceapi.detectAllFaces(imageCanvas as any, faceDetectionOptions)
      .withFaceLandmarks()
      .withFaceExpressions();

    // If no faces detected or results array is empty, create a default with neutral expression
    let resultsJson;
    if (!results || results.length === 0) {
      resultsJson = defaultNeutralExpression;
      console.log(`No faces detected in ${filePath}, using default neutral expression`);
    } else {
      // Use detected expressions
      resultsJson = results.map(res => ({ expressions: res.expressions }));
    }

    // Save results as JSON
    fs.writeFileSync(outputJsonPath, JSON.stringify(resultsJson, null, 2));
  } catch (error) {
    console.error(`Error processing image ${filePath}:`, error);
    try {
      fs.writeFileSync(outputJsonPath, JSON.stringify(defaultNeutralExpression, null, 2));
      console.log(`Error processing ${filePath}, created fallback neutral expression JSON`);
    } catch (writeError) {
      console.error(`Failed to write fallback JSON for ${filePath}:`, writeError);
    }
  }
};

const processSingleFolder = async (folderPath: string) => {
  const files = fs.readdirSync(folderPath);
  for (const file of files) {
    if (file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg')) {
      const filePath = path.join(folderPath, file);
      const outputJsonPath = path.join(folderPath, `${path.parse(file).name}.json`);
      const outputImagePath = path.join(folderPath, `${path.parse(file).name}_processed.jpg`);
      if (fs.existsSync(outputJsonPath)) {
        continue;
      }
      console.log(filePath);
      try {
        const stat = fs.statSync(filePath);
        if (stat.size === 0) {
          console.warn(`Empty image file in ${filePath}, using default neutral expression`);
          fs.writeFileSync(outputJsonPath, JSON.stringify(defaultNeutralExpression, null, 2));
          continue;
        }
        await processImage(filePath, outputJsonPath, outputImagePath);
      } catch (error) {
        console.error(`Unhandled error processing ${filePath}:`, error);
        try {
          fs.writeFileSync(outputJsonPath, JSON.stringify(defaultNeutralExpression, null, 2));
          console.log(`Fallback JSON written for ${filePath}`);
        } catch (writeError) {
          console.error(`Failed to write fallback JSON for ${filePath}:`, writeError);
        }
      }
    }
  }
};

const processAllImages = async () => {
  // Se SINGLE_FOLDER está definido, processa apenas essa pasta
  const singleFolder = process.env.SINGLE_FOLDER;
  if (singleFolder) {
    console.log(`Processing single folder: ${singleFolder}`);
    await processSingleFolder(singleFolder);
    return;
  }

  // Sem SINGLE_FOLDER: processa todas as subpastas (modo batch)
  const folders = fs.readdirSync(framesDir, { withFileTypes: true });
  for (const folder of folders) {
    if (folder.isDirectory()) {
      const folderPath = path.join(framesDir, folder.name);
      await processSingleFolder(folderPath);
    }
  }
};

await processAllImages();
