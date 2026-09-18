const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('@napi-rs/canvas');

async function generateIcoFromAngularEncoder(inputImagePath, outputIcoPath, targetSizes = [16, 24, 32, 48, 64, 128, 256, 512]) {
  const img = await loadImage(inputImagePath);
  
  // Create source canvas matching image dimensions
  const srcCanvas = createCanvas(img.width, img.height);
  const srcCtx = srcCanvas.getContext('2d');
  srcCtx.drawImage(img, 0, 0);

  const srcW = srcCanvas.width;
  const srcH = srcCanvas.height;

  const pngBuffers = [];

  for (const size of targetSizes) {
    const icoCanvas = createCanvas(size, size);
    const ctx = icoCanvas.getContext('2d');

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    ctx.clearRect(0, 0, size, size);

    const scale = Math.min(size / srcW, size / srcH);
    const drawW = Math.max(1, Math.round(srcW * scale));
    const drawH = Math.max(1, Math.round(srcH * scale));
    const drawX = Math.round((size - drawW) / 2);
    const drawY = Math.round((size - drawH) / 2);

    ctx.drawImage(srcCanvas, 0, 0, srcW, srcH, drawX, drawY, drawW, drawH);

    const pngBytes = await icoCanvas.toBuffer('image/png');
    pngBuffers.push({ size, bytes: new Uint8Array(pngBytes) });
  }

  const headerSize = 6;
  const dirEntrySize = 16;
  const numImages = pngBuffers.length;
  const directorySize = numImages * dirEntrySize;
  let totalPngBytes = 0;
  for (const entry of pngBuffers) {
    totalPngBytes += entry.bytes.length;
  }

  const totalSize = headerSize + directorySize + totalPngBytes;
  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  // ICONDIR (6 bytes)
  view.setUint16(0, 0, true);
  view.setUint16(2, 1, true);
  view.setUint16(4, numImages, true);

  let currentDataOffset = headerSize + directorySize;
  let dirOffset = headerSize;

  for (const entry of pngBuffers) {
    const bWidth = entry.size >= 256 ? 0 : entry.size;
    const bHeight = entry.size >= 256 ? 0 : entry.size;

    view.setUint8(dirOffset + 0, bWidth);
    view.setUint8(dirOffset + 1, bHeight);
    view.setUint8(dirOffset + 2, 0);
    view.setUint8(dirOffset + 3, 0);
    view.setUint16(dirOffset + 4, 1, true);
    view.setUint16(dirOffset + 6, 32, true);
    view.setUint32(dirOffset + 8, entry.bytes.length, true);
    view.setUint32(dirOffset + 12, currentDataOffset, true);

    bytes.set(entry.bytes, currentDataOffset);

    dirOffset += dirEntrySize;
    currentDataOffset += entry.bytes.length;
  }

  fs.writeFileSync(outputIcoPath, Buffer.from(buffer));
  console.log(`Successfully generated production ICO at ${outputIcoPath}`);
}

const inputPath = path.join(__dirname, 'test_images', 'extracted_icon.png');
const outputPath = path.join(__dirname, 'angular_generated.ico');

generateIcoFromAngularEncoder(inputPath, outputPath)
  .catch(err => {
    console.error('Error generating ICO:', err);
    process.exit(1);
  });
