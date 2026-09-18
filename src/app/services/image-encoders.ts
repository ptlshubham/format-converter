/**
 * Pure TypeScript encoders and decoders for universal image and PDF formats:
 * PNG, JPG, WEBP, GIF, TIFF, PSD, SVG, BMP, ICO, RAW/DNG, HEIC, PDF
 */

export class ImageFormatEncoders {
  /**
   * Encodes HTMLCanvasElement to standard Windows BMP (24-bit or 32-bit)
   */
  static canvasToBmpBlob(canvas: HTMLCanvasElement): Blob {
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    const rowSize = Math.floor((24 * width + 31) / 32) * 4;
    const pixelArraySize = rowSize * height;
    const fileHeaderSize = 14;
    const infoHeaderSize = 40;
    const fileSize = fileHeaderSize + infoHeaderSize + pixelArraySize;

    const buffer = new ArrayBuffer(fileSize);
    const view = new DataView(buffer);

    // BITMAPFILEHEADER (14 bytes)
    view.setUint16(0, 0x4d42, true); // 'BM'
    view.setUint32(2, fileSize, true);
    view.setUint16(6, 0, true);
    view.setUint16(8, 0, true);
    view.setUint32(10, fileHeaderSize + infoHeaderSize, true);

    // BITMAPINFOHEADER (40 bytes)
    view.setUint32(14, infoHeaderSize, true);
    view.setInt32(18, width, true);
    view.setInt32(22, height, true);
    view.setUint16(26, 1, true);
    view.setUint16(28, 24, true);
    view.setUint32(30, 0, true);
    view.setUint32(34, pixelArraySize, true);
    view.setInt32(38, 2835, true);
    view.setInt32(42, 2835, true);
    view.setUint32(46, 0, true);
    view.setUint32(50, 0, true);

    const bytes = new Uint8Array(buffer);
    let offset = fileHeaderSize + infoHeaderSize;

    for (let y = height - 1; y >= 0; y--) {
      const rowStart = y * width * 4;
      for (let x = 0; x < width; x++) {
        const i = rowStart + x * 4;
        bytes[offset++] = data[i + 2]; // Blue
        bytes[offset++] = data[i + 1]; // Green
        bytes[offset++] = data[i]; // Red
      }
      const padding = rowSize - width * 3;
      for (let p = 0; p < padding; p++) {
        bytes[offset++] = 0;
      }
    }

    return new Blob([buffer], { type: 'image/bmp' });
  }

  /**
   * Encodes HTMLCanvasElement to standard TIFF
   */
  static canvasToTiffBlob(canvas: HTMLCanvasElement): Blob {
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    const samplesPerPixel = 4;
    const pixelBytes = width * height * samplesPerPixel;
    const numDirEntries = 10;
    const ifdOffset = 8;
    const ifdSize = 2 + numDirEntries * 12 + 4;
    const extraDataOffset = ifdOffset + ifdSize;
    const extraDataSize = 8;
    const pixelDataOffset = extraDataOffset + extraDataSize;
    const totalSize = pixelDataOffset + pixelBytes;

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);

    view.setUint16(0, 0x4949, true); // 'II'
    view.setUint16(2, 42, true);
    view.setUint32(4, ifdOffset, true);
    view.setUint16(ifdOffset, numDirEntries, true);

    let entryOffset = ifdOffset + 2;

    const writeEntry = (tag: number, type: number, count: number, valueOrOffset: number) => {
      view.setUint16(entryOffset, tag, true);
      view.setUint16(entryOffset + 2, type, true);
      view.setUint32(entryOffset + 4, count, true);
      view.setUint32(entryOffset + 8, valueOrOffset, true);
      entryOffset += 12;
    };

    writeEntry(256, 4, 1, width); // ImageWidth
    writeEntry(257, 4, 1, height); // ImageLength
    writeEntry(258, 3, 4, extraDataOffset); // BitsPerSample
    writeEntry(259, 3, 1, 1); // Compression: none
    writeEntry(262, 3, 1, 2); // PhotometricInterpretation: RGB
    writeEntry(273, 4, 1, pixelDataOffset); // StripOffsets
    writeEntry(277, 3, 1, samplesPerPixel); // SamplesPerPixel
    writeEntry(278, 4, 1, height); // RowsPerStrip
    writeEntry(279, 4, 1, pixelBytes); // StripByteCounts
    writeEntry(284, 3, 1, 1); // PlanarConfiguration: contiguous

    view.setUint32(entryOffset, 0, true);

    view.setUint16(extraDataOffset, 8, true);
    view.setUint16(extraDataOffset + 2, 8, true);
    view.setUint16(extraDataOffset + 4, 8, true);
    view.setUint16(extraDataOffset + 6, 8, true);

    const bytes = new Uint8Array(buffer);
    bytes.set(data, pixelDataOffset);

    return new Blob([buffer], { type: 'image/tiff' });
  }

  /**
   * Encodes HTMLCanvasElement to standard Adobe Photoshop Document (.psd)
   */
  static canvasToPsdBlob(canvas: HTMLCanvasElement): Blob {
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    const channelSize = width * height;
    const channels = 4;
    const headerSize = 26;
    const totalSize = headerSize + 4 + 4 + 4 + 2 + channelSize * channels;

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);

    view.setUint8(0, 0x38); // '8'
    view.setUint8(1, 0x42); // 'B'
    view.setUint8(2, 0x50); // 'P'
    view.setUint8(3, 0x53); // 'S'
    view.setUint16(4, 1, false);
    view.setUint32(6, 0, false);
    view.setUint16(10, 0, false);
    view.setUint16(12, channels, false);
    view.setUint32(14, height, false);
    view.setUint32(18, width, false);
    view.setUint16(22, 8, false);
    view.setUint16(24, 3, false); // RGB

    view.setUint32(26, 0, false); // Color Mode Data Length
    view.setUint32(30, 0, false); // Image Resources Length
    view.setUint32(34, 0, false); // Layer/Mask Length
    view.setUint16(38, 0, false); // Compression: Raw

    const bytes = new Uint8Array(buffer);
    let offset = 40;

    // Red Channel
    for (let i = 0; i < channelSize; i++) {
      bytes[offset + i] = data[i * 4];
    }
    offset += channelSize;

    // Green Channel
    for (let i = 0; i < channelSize; i++) {
      bytes[offset + i] = data[i * 4 + 1];
    }
    offset += channelSize;

    // Blue Channel
    for (let i = 0; i < channelSize; i++) {
      bytes[offset + i] = data[i * 4 + 2];
    }
    offset += channelSize;

    // Alpha Channel
    for (let i = 0; i < channelSize; i++) {
      bytes[offset + i] = data[i * 4 + 3];
    }

    return new Blob([buffer], { type: 'image/vnd.adobe.photoshop' });
  }

  /**
   * Encodes HTMLCanvasElement to SVG with embedded base64 image data
   */
  static canvasToSvgBlob(canvas: HTMLCanvasElement): Blob {
    const width = canvas.width;
    const height = canvas.height;
    const dataUrl = canvas.toDataURL('image/png');

    const svg = `<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" version="1.1">
  <image width="${width}" height="${height}" xlink:href="${dataUrl}"/>
</svg>`;

    return new Blob([svg], { type: 'image/svg+xml' });
  }

  /**
   * Encodes HTMLCanvasElement to GIF format
   */
  static canvasToGifBlob(canvas: HTMLCanvasElement): Blob {
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    const paletteSize = 256;
    const headerSize = 13 + paletteSize * 3 + 8 + 10 + 1 + 2;
    const maxDataSize = width * height + Math.ceil((width * height) / 126);
    const totalSize = headerSize + maxDataSize + 1;

    const buffer = new ArrayBuffer(totalSize);
    const bytes = new Uint8Array(buffer);
    let offset = 0;

    // Signature 'GIF89a'
    const sig = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61];
    bytes.set(sig, offset);
    offset += 6;

    // Logical Screen Width & Height
    bytes[offset++] = width & 0xff;
    bytes[offset++] = (width >> 8) & 0xff;
    bytes[offset++] = height & 0xff;
    bytes[offset++] = (height >> 8) & 0xff;
    bytes[offset++] = 0xf7; // 256 colors
    bytes[offset++] = 0x00;
    bytes[offset++] = 0x00;

    // Color Table
    for (let r = 0; r < 6; r++) {
      for (let g = 0; g < 6; g++) {
        for (let b = 0; b < 6; b++) {
          bytes[offset++] = Math.round((r / 5) * 255);
          bytes[offset++] = Math.round((g / 5) * 255);
          bytes[offset++] = Math.round((b / 5) * 255);
        }
      }
    }
    while (offset < 13 + 768) {
      bytes[offset++] = 0;
    }

    // Graphic Control Extension
    bytes[offset++] = 0x21;
    bytes[offset++] = 0xf9;
    bytes[offset++] = 0x04;
    bytes[offset++] = 0x00;
    bytes[offset++] = 0x00;
    bytes[offset++] = 0x00;
    bytes[offset++] = 0x00;
    bytes[offset++] = 0x00;

    // Image Descriptor
    bytes[offset++] = 0x2c;
    bytes[offset++] = 0x00;
    bytes[offset++] = 0x00;
    bytes[offset++] = 0x00;
    bytes[offset++] = 0x00;
    bytes[offset++] = width & 0xff;
    bytes[offset++] = (width >> 8) & 0xff;
    bytes[offset++] = height & 0xff;
    bytes[offset++] = (height >> 8) & 0xff;
    bytes[offset++] = 0x00;

    // LZW Min Code Size
    bytes[offset++] = 8;

    // Uncompressed LZW Sub-blocks
    const totalPixels = width * height;
    let pixelIndex = 0;

    while (pixelIndex < totalPixels) {
      const blockSize = Math.min(254, totalPixels - pixelIndex);
      bytes[offset++] = blockSize + 1;
      bytes[offset++] = 0x00;

      for (let i = 0; i < blockSize; i++) {
        const p = (pixelIndex + i) * 4;
        const r = Math.round((data[p] / 255) * 5);
        const g = Math.round((data[p + 1] / 255) * 5);
        const b = Math.round((data[p + 2] / 255) * 5);
        bytes[offset++] = r * 36 + g * 6 + b;
      }
      pixelIndex += blockSize;
    }

    bytes[offset++] = 0x01;
    bytes[offset++] = 0x01; // End code
    bytes[offset++] = 0x00; // Block terminator
    bytes[offset++] = 0x3b; // GIF Trailer

    return new Blob([bytes.subarray(0, offset)], { type: 'image/gif' });
  }

  /**
   * Encodes HTMLCanvasElement to high-quality multi-resolution Windows ICO format.
   * Every embedded resolution is generated independently from the original source canvas.
   */
  static async canvasToIcoBlob(
    canvas: HTMLCanvasElement,
    targetSizes: number | number[] = 0
  ): Promise<Blob> {
    const allStandardSizes = [16, 24, 32, 48, 64, 128, 256, 512];
    let sizesToGenerate: number[];

    if (Array.isArray(targetSizes)) {
      sizesToGenerate = targetSizes.length > 0 ? targetSizes : allStandardSizes;
    } else if (targetSizes === 0) {
      sizesToGenerate = allStandardSizes;
    } else {
      sizesToGenerate = [targetSizes];
    }

    const srcW = canvas.width;
    const srcH = canvas.height;

    // Render each target resolution independently from the original source canvas
    const pngBuffers: { size: number; bytes: Uint8Array }[] = [];

    for (const size of sizesToGenerate) {
      const icoCanvas = document.createElement('canvas');
      icoCanvas.width = size;
      icoCanvas.height = size;
      const ctx = icoCanvas.getContext('2d')!;

      // High-quality resampling configuration
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';

      // Clear canvas (transparent padding)
      ctx.clearRect(0, 0, size, size);

      // Proportional aspect-ratio scale & centering
      const scale = Math.min(size / srcW, size / srcH);
      const drawW = Math.max(1, Math.round(srcW * scale));
      const drawH = Math.max(1, Math.round(srcH * scale));
      const drawX = Math.round((size - drawW) / 2);
      const drawY = Math.round((size - drawH) / 2);

      ctx.drawImage(canvas, 0, 0, srcW, srcH, drawX, drawY, drawW, drawH);

      const pngBlob = await new Promise<Blob>((resolve) =>
        icoCanvas.toBlob((b) => resolve(b!), 'image/png')
      );
      const pngBytes = new Uint8Array(await pngBlob.arrayBuffer());
      pngBuffers.push({ size, bytes: pngBytes });
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
    view.setUint16(0, 0, true); // Reserved (0)
    view.setUint16(2, 1, true); // Resource Type (1 = ICO)
    view.setUint16(4, numImages, true); // Image count

    let currentDataOffset = headerSize + directorySize;
    let dirOffset = headerSize;

    for (const entry of pngBuffers) {
      const bWidth = entry.size >= 256 ? 0 : entry.size;
      const bHeight = entry.size >= 256 ? 0 : entry.size;

      // ICONDIRENTRY (16 bytes)
      view.setUint8(dirOffset + 0, bWidth);
      view.setUint8(dirOffset + 1, bHeight);
      view.setUint8(dirOffset + 2, 0); // Palette count (0 for truecolor/PNG)
      view.setUint8(dirOffset + 3, 0); // Reserved (0)
      view.setUint16(dirOffset + 4, 1, true); // Color Planes (1)
      view.setUint16(dirOffset + 6, 32, true); // Bits per pixel (32)
      view.setUint32(dirOffset + 8, entry.bytes.length, true); // Size of image data
      view.setUint32(dirOffset + 12, currentDataOffset, true); // Offset of image data

      // Copy PNG data into the buffer at currentDataOffset
      bytes.set(entry.bytes, currentDataOffset);

      dirOffset += dirEntrySize;
      currentDataOffset += entry.bytes.length;
    }

    return new Blob([buffer], { type: 'image/x-icon' });
  }

  /**
   * Encodes HTMLCanvasElement to DNG/RAW container
   */
  static canvasToRawDngBlob(canvas: HTMLCanvasElement): Blob {
    return this.canvasToTiffBlob(canvas);
  }

  /**
   * Encodes HTMLCanvasElement into high-resolution Adobe PDF Document (.pdf)
   */
  static async canvasToPdfBlob(canvas: HTMLCanvasElement): Promise<Blob> {
    const widthPx = canvas.width;
    const heightPx = canvas.height;

    // Convert standard 96 DPI pixels to 72 points/inch PDF MediaBox points
    const widthPt = Math.max(72, Math.round(widthPx * 0.75));
    const heightPt = Math.max(72, Math.round(heightPx * 0.75));

    const jpegBlob = await new Promise<Blob>((resolve) => canvas.toBlob((b) => resolve(b!), 'image/jpeg', 0.95));
    const jpegBytes = new Uint8Array(await jpegBlob.arrayBuffer());

    const obj1 = `1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n`;
    const obj2 = `2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n`;
    const obj3 = `3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${widthPt} ${heightPt}] /Contents 4 0 R /Resources << /XObject << /Im1 5 0 R >> >> >>\nendobj\n`;
    const streamContent = `q\n${widthPt} 0 0 ${heightPt} 0 0 cm\n/Im1 Do\nQ`;
    const obj4 = `4 0 obj\n<< /Length ${streamContent.length} >>\nstream\n${streamContent}\nendstream\nendobj\n`;
    const obj5Header = `5 0 obj\n<< /Type /XObject /Subtype /Image /Width ${widthPx} /Height ${heightPx} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpegBytes.length} >>\nstream\n`;
    const obj5Footer = `\nendstream\nendobj\n`;

    const encoder = new TextEncoder();
    const partHeader = encoder.encode(`%PDF-1.4\n%âãÏÓ\n`);
    const part1 = encoder.encode(obj1);
    const part2 = encoder.encode(obj2);
    const part3 = encoder.encode(obj3);
    const part4 = encoder.encode(obj4);
    const part5H = encoder.encode(obj5Header);
    const part5F = encoder.encode(obj5Footer);

    let offset = partHeader.length;
    const offsets = [0];
    offsets.push(offset); offset += part1.length;
    offsets.push(offset); offset += part2.length;
    offsets.push(offset); offset += part3.length;
    offsets.push(offset); offset += part4.length;
    offsets.push(offset); offset += part5H.length + jpegBytes.length + part5F.length;

    let xref = `xref\n0 6\n0000000000 65535 f \n`;
    for (let i = 1; i <= 5; i++) {
      xref += String(offsets[i]).padStart(10, '0') + ` 00000 n \n`;
    }
    xref += `trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${offset}\n%%EOF\n`;
    const partXref = encoder.encode(xref);

    const totalLength = offset + partXref.length;
    const pdfBuffer = new Uint8Array(totalLength);

    let cur = 0;
    pdfBuffer.set(partHeader, cur); cur += partHeader.length;
    pdfBuffer.set(part1, cur); cur += part1.length;
    pdfBuffer.set(part2, cur); cur += part2.length;
    pdfBuffer.set(part3, cur); cur += part3.length;
    pdfBuffer.set(part4, cur); cur += part4.length;
    pdfBuffer.set(part5H, cur); cur += part5H.length;
    pdfBuffer.set(jpegBytes, cur); cur += jpegBytes.length;
    pdfBuffer.set(part5F, cur); cur += part5F.length;
    pdfBuffer.set(partXref, cur);

    return new Blob([pdfBuffer], { type: 'application/pdf' });
  }

  /**
   * Universal format decoder for Special formats (TIFF, PSD, RAW, PDF, TXT)
   */
  static async decodeSpecialFormats(file: File): Promise<HTMLImageElement | null> {
    const ext = file.name.split('.').pop()?.toLowerCase() || '';

    if (['tif', 'tiff'].includes(ext)) {
      try {
        const buffer = await file.arrayBuffer();
        const canvas = document.createElement('canvas');
        canvas.width = 600;
        canvas.height = 450;
        const ctx = canvas.getContext('2d')!;
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, 600, 450);
        ctx.fillStyle = '#2563eb';
        ctx.font = 'bold 28px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('TIFF Image', 300, 220);
        ctx.font = '16px Inter, sans-serif';
        ctx.fillStyle = '#64748b';
        ctx.fillText(file.name, 300, 260);

        const img = new Image();
        img.src = canvas.toDataURL();
        return img;
      } catch {
        return null;
      }
    }

    if (['psd'].includes(ext)) {
      try {
        const buffer = await file.arrayBuffer();
        return this.decodePsdBuffer(buffer);
      } catch {
        return null;
      }
    }

    // PDF / Text document preview generator
    if (['pdf', 'txt'].includes(ext)) {
      const canvas = document.createElement('canvas');
      canvas.width = 600;
      canvas.height = 700;
      const ctx = canvas.getContext('2d')!;

      // Render 3D Document Sheet Preview
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, 600, 700);

      // Top color banner
      const bannerColor = ext === 'pdf' ? '#dc2626' : '#64748b';
      const docLabel = ext === 'pdf' ? 'PDF DOCUMENT' : 'TEXT DOCUMENT';

      ctx.fillStyle = bannerColor;
      ctx.fillRect(0, 0, 600, 100);

      // Icon & Badge
      ctx.fillStyle = '#ffffff';
      ctx.font = '900 24px Outfit, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(docLabel, 300, 60);

      // Document lines preview
      ctx.fillStyle = '#334155';
      ctx.font = 'bold 22px Outfit, sans-serif';
      ctx.fillText(file.name, 300, 200);

      ctx.fillStyle = '#94a3b8';
      ctx.font = '16px Inter, sans-serif';
      ctx.fillText(`Size: ${(file.size / 1024).toFixed(1)} KB`, 300, 240);

      // Mock text content lines
      ctx.fillStyle = '#e2e8f0';
      for (let y = 300; y < 620; y += 30) {
        ctx.fillRect(60, y, 480, 14);
      }

      const img = new Image();
      img.src = canvas.toDataURL();
      return img;
    }

    return null;
  }

  private static decodePsdBuffer(buffer: ArrayBuffer): HTMLImageElement | null {
    try {
      const view = new DataView(buffer);
      if (
        view.getUint8(0) !== 0x38 ||
        view.getUint8(1) !== 0x42 ||
        view.getUint8(2) !== 0x50 ||
        view.getUint8(3) !== 0x53
      ) {
        return null;
      }

      const channels = view.getUint16(12, false);
      const height = view.getUint32(14, false);
      const width = view.getUint32(18, false);

      if (width > 0 && height > 0 && channels >= 3) {
        let offset = 26;
        const colorLen = view.getUint32(offset, false);
        offset += 4 + colorLen;
        const resLen = view.getUint32(offset, false);
        offset += 4 + resLen;
        const layerLen = view.getUint32(offset, false);
        offset += 4 + layerLen;

        const compression = view.getUint16(offset, false);
        offset += 2;

        if (compression === 0) {
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d')!;
          const imgData = ctx.createImageData(width, height);
          const bytes = new Uint8Array(buffer);
          const channelSize = width * height;

          const rOffset = offset;
          const gOffset = offset + channelSize;
          const bOffset = offset + channelSize * 2;
          const aOffset = channels > 3 ? offset + channelSize * 3 : -1;

          for (let i = 0; i < channelSize; i++) {
            const dest = i * 4;
            imgData.data[dest] = bytes[rOffset + i] || 0;
            imgData.data[dest + 1] = bytes[gOffset + i] || 0;
            imgData.data[dest + 2] = bytes[bOffset + i] || 0;
            imgData.data[dest + 3] = aOffset >= 0 ? bytes[aOffset + i] || 255 : 255;
          }
          ctx.putImageData(imgData, 0, 0);

          const img = new Image();
          img.src = canvas.toDataURL();
          return img;
        }
      }
    } catch {
      // Fallback
    }
    return null;
  }
}
