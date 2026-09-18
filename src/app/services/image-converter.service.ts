import { Injectable } from '@angular/core';
import JSZip from 'jszip';
import {
  ConvertSettings,
  FormatOption,
  ImageItem,
  SupportedFormat,
} from '../models/image-item.model';
import { ImageFormatEncoders } from './image-encoders';

export const SUPPORTED_FORMATS: FormatOption[] = [
  {
    id: 'webp',
    label: 'WEBP',
    ext: '.webp',
    mimeType: 'image/webp',
    description: 'Modern Web Picture Format (High Compression)',
  },
  {
    id: 'jpg',
    label: 'JPG / JPEG',
    ext: '.jpg',
    mimeType: 'image/jpeg',
    description: 'Joint Photographic Experts Group',
  },
  {
    id: 'png',
    label: 'PNG',
    ext: '.png',
    mimeType: 'image/png',
    description: 'Portable Network Graphics (Transparency)',
  },
  {
    id: 'pdf',
    label: 'PDF Document',
    ext: '.pdf',
    mimeType: 'application/pdf',
    description: 'Adobe Portable Document Format (PDF)',
  },
  {
    id: 'gif',
    label: 'GIF',
    ext: '.gif',
    mimeType: 'image/gif',
    description: 'Graphics Interchange Format',
  },
  {
    id: 'tif',
    label: 'TIFF / TIF',
    ext: '.tif',
    mimeType: 'image/tiff',
    description: 'Tagged Image File Format (Print Quality)',
  },
  {
    id: 'psd',
    label: 'PSD',
    ext: '.psd',
    mimeType: 'image/vnd.adobe.photoshop',
    description: 'Adobe Photoshop Document',
  },
  {
    id: 'svg',
    label: 'SVG',
    ext: '.svg',
    mimeType: 'image/svg+xml',
    description: 'Scalable Vector Graphics',
  },
  {
    id: 'bmp',
    label: 'BMP',
    ext: '.bmp',
    mimeType: 'image/bmp',
    description: 'Bitmap Image File',
  },
  {
    id: 'ico',
    label: 'ICO',
    ext: '.ico',
    mimeType: 'image/x-icon',
    description: 'Windows Icon File',
  },
  {
    id: 'raw',
    label: 'RAW / DNG',
    ext: '.dng',
    mimeType: 'image/x-adobe-dng',
    description: 'Digital Negative / Camera RAW Container',
  },
  {
    id: 'heic',
    label: 'HEIC',
    ext: '.heic',
    mimeType: 'image/heic',
    description: 'High Efficiency Image Container',
  },
];

@Injectable({
  providedIn: 'root',
})
export class ImageConverterService {
  /**
   * Loads any file into an HTMLImageElement, resolving special formats (TIFF, PSD, RAW, PDF)
   */
  async loadImage(file: File): Promise<{ img: HTMLImageElement; width: number; height: number }> {
    const specialImg = await ImageFormatEncoders.decodeSpecialFormats(file);
    if (specialImg) {
      return {
        img: specialImg,
        width: specialImg.naturalWidth || specialImg.width,
        height: specialImg.naturalHeight || specialImg.height,
      };
    }

    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve({
          img,
          width: img.naturalWidth || img.width,
          height: img.naturalHeight || img.height,
        });
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error(`Failed to load file: ${file.name}`));
      };
      img.src = url;
    });
  }

  /**
   * Renders the image with its rotation applied onto an HTMLCanvasElement
   */
  createTransformedCanvas(
    img: HTMLImageElement,
    rotation: number,
    preserveTransparency: boolean,
    targetFormat: SupportedFormat
  ): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    const rad = ((rotation % 360) * Math.PI) / 180;
    const isSideways = rotation % 180 !== 0;

    const naturalWidth = img.naturalWidth || img.width || 600;
    const naturalHeight = img.naturalHeight || img.height || 400;

    canvas.width = isSideways ? naturalHeight : naturalWidth;
    canvas.height = isSideways ? naturalWidth : naturalHeight;

    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;

    const supportsTransparency = ['png', 'webp', 'svg', 'ico', 'gif', 'tif', 'psd', 'pdf'].includes(
      targetFormat
    );

    if (!supportsTransparency || !preserveTransparency) {
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate(rad);
    ctx.drawImage(img, -naturalWidth / 2, -naturalHeight / 2);
    ctx.restore();

    return canvas;
  }

  /**
   * Converts an ImageItem/DocumentItem to the target format with specified settings
   */
  async convertSingleItem(
    item: ImageItem,
    settings: ConvertSettings
  ): Promise<{ blob: Blob; url: string; newName: string; size: number }> {
    const { img } = await this.loadImage(item.file);
    const canvas = this.createTransformedCanvas(
      img,
      item.rotation,
      settings.preserveTransparency,
      settings.targetFormat
    );

    let blob: Blob;

    switch (settings.targetFormat) {
      case 'webp':
        blob = await new Promise<Blob>((resolve) =>
          canvas.toBlob((b) => resolve(b!), 'image/webp', settings.quality)
        );
        break;

      case 'jpg':
        blob = await new Promise<Blob>((resolve) =>
          canvas.toBlob((b) => resolve(b!), 'image/jpeg', settings.quality)
        );
        break;

      case 'png':
        blob = await new Promise<Blob>((resolve) =>
          canvas.toBlob((b) => resolve(b!), 'image/png')
        );
        break;

      case 'pdf':
        blob = await ImageFormatEncoders.canvasToPdfBlob(canvas);
        break;

      case 'gif':
        blob = ImageFormatEncoders.canvasToGifBlob(canvas);
        break;

      case 'tif':
        blob = ImageFormatEncoders.canvasToTiffBlob(canvas);
        break;

      case 'psd':
        blob = ImageFormatEncoders.canvasToPsdBlob(canvas);
        break;

      case 'svg':
        blob = ImageFormatEncoders.canvasToSvgBlob(canvas);
        break;

      case 'bmp':
        blob = ImageFormatEncoders.canvasToBmpBlob(canvas);
        break;

      case 'ico':
        blob = await ImageFormatEncoders.canvasToIcoBlob(canvas, settings.icoSize);
        break;

      case 'raw':
      case 'heic':
        blob = ImageFormatEncoders.canvasToRawDngBlob(canvas);
        break;

      default:
        blob = await new Promise<Blob>((resolve) =>
          canvas.toBlob((b) => resolve(b!), 'image/jpeg', 0.92)
        );
        break;
    }

    const baseName = item.name.substring(0, item.name.lastIndexOf('.')) || item.name;
    const formatMeta = SUPPORTED_FORMATS.find((f) => f.id === settings.targetFormat);
    const ext = formatMeta ? formatMeta.ext : `.${settings.targetFormat}`;
    const newName = `${baseName}${ext}`;
    const url = URL.createObjectURL(blob);

    return {
      blob,
      url,
      newName,
      size: blob.size,
    };
  }

  /**
   * Bundles converted images and documents into a ZIP archive
   */
  async createZipArchive(items: ImageItem[]): Promise<Blob> {
    const zip = new JSZip();

    for (const item of items) {
      if (item.convertedBlob && item.convertedName) {
        zip.file(item.convertedName, item.convertedBlob);
      }
    }

    return await zip.generateAsync({
      type: 'blob',
      compression: 'DEFLATE',
      compressionOptions: { level: 6 },
    });
  }

  /**
   * Helper to trigger browser download of a Blob
   */
  downloadBlob(blob: Blob, filename: string): void {
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }
}
