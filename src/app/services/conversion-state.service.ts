import { computed, inject, Injectable, signal } from '@angular/core';
import {
  ConvertSettings,
  FormatOption,
  ImageItem,
  SupportedFormat,
} from '../models/image-item.model';
import { ImageConverterService, SUPPORTED_FORMATS } from './image-converter.service';

export type AppStep = 'landing' | 'workspace' | 'result';
export type AppTool = 'converter' | 'bg-remover';

@Injectable({
  providedIn: 'root',
})
export class ConversionStateService {
  private converter = inject(ImageConverterService);

  readonly activeTool = signal<AppTool>('converter');
  readonly step = signal<AppStep>('landing');
  readonly images = signal<ImageItem[]>([]);
  readonly targetFormat = signal<SupportedFormat>('jpg');
  readonly quality = signal<number>(0.85);
  readonly preserveTransparency = signal<boolean>(true);
  readonly icoSize = signal<number>(0);
  readonly isConverting = signal<boolean>(false);
  readonly overallProgress = signal<number>(0);
  readonly currentConvertingIndex = signal<number>(0);

  readonly supportedFormats = SUPPORTED_FORMATS;

  readonly currentFormatOption = computed<FormatOption>(() => {
    const fmt = this.targetFormat();
    return SUPPORTED_FORMATS.find((f) => f.id === fmt) || SUPPORTED_FORMATS[0];
  });

  readonly totalImagesCount = computed(() => this.images().length);

  readonly convertedImagesCount = computed(
    () => this.images().filter((img) => img.status === 'done').length
  );

  readonly totalOriginalSize = computed(() =>
    this.images().reduce((acc, curr) => acc + curr.originalSize, 0)
  );

  readonly totalConvertedSize = computed(() =>
    this.images().reduce((acc, curr) => acc + (curr.convertedSize || 0), 0)
  );

  /**
   * Live estimated output size computed dynamically as quality, format, or images change
   */
  readonly estimatedOutputSize = computed(() => {
    const images = this.images();
    if (!images.length) return 0;
    return images.reduce((acc, item) => acc + this.getEstimatedSizeForItem(item), 0);
  });

  /**
   * Live estimated percentage reduction / savings
   */
  readonly estimatedSavingsPercent = computed(() => {
    const orig = this.totalOriginalSize();
    const est = this.estimatedOutputSize();
    if (!orig || !est || est >= orig) return 0;
    const savings = Math.round(((orig - est) / orig) * 100);
    return Math.min(99, Math.max(0, savings));
  });

  /**
   * Calculate live estimated output size for an individual item based on input format, target format, and quality.
   */
  getEstimatedSizeForItem(itemOrSize: ImageItem | number, formatHint?: string): number {
    let origSize = 0;
    let inFmt = 'JPG';

    if (typeof itemOrSize === 'object' && itemOrSize !== null) {
      origSize = itemOrSize.originalSize || 0;
      inFmt = (itemOrSize.originalFormat || 'JPG').toUpperCase();
    } else {
      origSize = itemOrSize || 0;
      if (formatHint) {
        inFmt = formatHint.toUpperCase();
      }
    }

    if (!origSize || origSize <= 0) return 0;

    const targetFmt = this.targetFormat();
    const q = Math.max(0.1, Math.min(1.0, this.quality())); // 0.1 to 1.0

    // Categorize input format into compression families
    const isLosslessPng = inFmt === 'PNG';
    const isJpgLike = ['JPG', 'JPEG', 'JPE', 'JFIF'].includes(inFmt);
    const isUncompressedOrRaw = ['BMP', 'TIF', 'TIFF', 'PSD', 'RAW', 'CR2', 'NEF', 'ARW', 'DNG', 'ORF'].includes(
      inFmt
    );

    let estimatedBytes = origSize;

    switch (targetFmt) {
      case 'webp': {
        // WebP VP8 compression is remarkably efficient.
        // Lossy WebP from uncompressed/PNG achieves ~8%-12% of PNG raster file size.
        // Lossy WebP from already-compressed JPEG achieves ~30%-45% of JPEG size.
        if (isLosslessPng) {
          // e.g. 300KB PNG -> ~28KB at q=0.85; 221KB PNG -> ~20-25KB
          const ratio = 0.04 + 0.075 * Math.pow(q, 2.2);
          estimatedBytes = Math.round(origSize * ratio);
        } else if (isJpgLike) {
          // e.g. 300KB JPG -> ~100KB at q=0.85
          const ratio = 0.12 + 0.28 * Math.pow(q, 1.8);
          estimatedBytes = Math.round(origSize * ratio);
        } else if (isUncompressedOrRaw) {
          // e.g. 5MB BMP/RAW -> ~100KB WebP
          const ratio = 0.015 + 0.04 * Math.pow(q, 2.0);
          estimatedBytes = Math.round(origSize * ratio);
        } else if (inFmt === 'WEBP') {
          const ratio = 0.35 + 0.55 * Math.pow(q, 1.5);
          estimatedBytes = Math.round(origSize * ratio);
        } else {
          const ratio = 0.06 + 0.14 * Math.pow(q, 2.0);
          estimatedBytes = Math.round(origSize * ratio);
        }
        break;
      }

      case 'jpg': {
        // JPEG DCT lossy compression
        if (isLosslessPng) {
          // e.g. 300KB PNG -> ~45KB-55KB JPEG at q=0.85
          const ratio = 0.06 + 0.14 * Math.pow(q, 1.8);
          estimatedBytes = Math.round(origSize * ratio);
        } else if (isJpgLike) {
          // Re-compressing JPEG at chosen quality
          const ratio = 0.22 + 0.65 * Math.pow(q, 1.6);
          estimatedBytes = Math.round(origSize * ratio);
        } else if (isUncompressedOrRaw) {
          const ratio = 0.02 + 0.06 * Math.pow(q, 1.8);
          estimatedBytes = Math.round(origSize * ratio);
        } else if (inFmt === 'WEBP') {
          const ratio = 0.7 + 0.85 * Math.pow(q, 1.5);
          estimatedBytes = Math.round(origSize * ratio);
        } else {
          const ratio = 0.15 + 0.35 * Math.pow(q, 1.8);
          estimatedBytes = Math.round(origSize * ratio);
        }
        break;
      }

      case 'png': {
        // PNG is lossless Deflate compression
        if (isLosslessPng) {
          estimatedBytes = Math.round(origSize * 0.94);
        } else if (isJpgLike) {
          // Decompressing DCT lossy image to lossless PNG inflates data size by ~2.2x
          estimatedBytes = Math.round(origSize * 2.2);
        } else if (inFmt === 'WEBP') {
          estimatedBytes = Math.round(origSize * 2.5);
        } else if (isUncompressedOrRaw) {
          estimatedBytes = Math.round(origSize * 0.28);
        } else {
          estimatedBytes = Math.round(origSize * 1.15);
        }
        break;
      }

      case 'pdf': {
        // PDF wrapper around a high-quality JPEG canvas stream (~1600 bytes header/xref)
        if (isLosslessPng) {
          estimatedBytes = Math.round(origSize * 0.18 + 1600);
        } else if (isJpgLike) {
          estimatedBytes = Math.round(origSize * 0.92 + 1600);
        } else if (isUncompressedOrRaw) {
          estimatedBytes = Math.round(origSize * 0.05 + 1600);
        } else {
          estimatedBytes = Math.round(origSize * 0.75 + 1600);
        }
        break;
      }

      case 'ico': {
        const size = this.icoSize();
        if (size === 0) {
          // Multi-resolution ICO (16, 24, 32, 48, 64, 128, 256, 512)
          const sizes = [16, 24, 32, 48, 64, 128, 256, 512];
          const totalPixels = sizes.reduce((sum, s) => sum + s * s, 0);
          estimatedBytes = Math.round(totalPixels * 0.42 + 6 + sizes.length * 16);
        } else {
          estimatedBytes = Math.round(size * size * 0.42 + 400);
        }
        break;
      }

      case 'gif': {
        // 256-color indexed palette
        if (isLosslessPng) {
          estimatedBytes = Math.round(origSize * 0.45);
        } else if (isJpgLike) {
          estimatedBytes = Math.round(origSize * 0.75);
        } else if (isUncompressedOrRaw) {
          estimatedBytes = Math.round(origSize * 0.12);
        } else {
          estimatedBytes = Math.round(origSize * 0.6);
        }
        break;
      }

      case 'svg': {
        // Base64 encoded PNG in SVG container: PNG * 1.37 + 350
        const estPng = isLosslessPng
          ? origSize * 0.94
          : isJpgLike
          ? origSize * 2.2
          : isUncompressedOrRaw
          ? origSize * 0.28
          : origSize * 1.15;
        estimatedBytes = Math.round(estPng * 1.37 + 350);
        break;
      }

      case 'bmp': {
        // 24-bit uncompressed RGB
        if (inFmt === 'BMP') {
          estimatedBytes = origSize;
        } else if (isJpgLike || inFmt === 'WEBP') {
          estimatedBytes = Math.round(origSize * 6.0);
        } else if (isLosslessPng) {
          estimatedBytes = Math.round(origSize * 3.8);
        } else if (isUncompressedOrRaw) {
          estimatedBytes = Math.round(origSize * 0.95);
        } else {
          estimatedBytes = Math.round(origSize * 4.5);
        }
        break;
      }

      case 'tif': {
        // 32-bit uncompressed RGBA TIFF
        if (inFmt === 'TIF' || inFmt === 'TIFF') {
          estimatedBytes = origSize;
        } else if (isJpgLike || inFmt === 'WEBP') {
          estimatedBytes = Math.round(origSize * 7.5);
        } else if (isLosslessPng) {
          estimatedBytes = Math.round(origSize * 4.8);
        } else if (isUncompressedOrRaw) {
          estimatedBytes = Math.round(origSize * 1.05);
        } else {
          estimatedBytes = Math.round(origSize * 5.5);
        }
        break;
      }

      case 'psd': {
        // Photoshop Document header + channels
        if (inFmt === 'PSD') {
          estimatedBytes = origSize;
        } else if (isLosslessPng) {
          estimatedBytes = Math.round(origSize * 4.5 + 500);
        } else if (isJpgLike || inFmt === 'WEBP') {
          estimatedBytes = Math.round(origSize * 6.5 + 500);
        } else if (isUncompressedOrRaw) {
          estimatedBytes = Math.round(origSize * 0.9 + 500);
        } else {
          estimatedBytes = Math.round(origSize * 4.0 + 500);
        }
        break;
      }

      case 'raw':
      case 'heic': {
        if (isUncompressedOrRaw) {
          estimatedBytes = Math.round(origSize * 0.95);
        } else if (isJpgLike || inFmt === 'WEBP') {
          estimatedBytes = Math.round(origSize * 3.5);
        } else {
          estimatedBytes = Math.round(origSize * 2.0);
        }
        break;
      }

      default: {
        const ratio = 0.15 + 0.35 * Math.pow(q, 1.8);
        estimatedBytes = Math.round(origSize * ratio);
        break;
      }
    }

    return Math.max(512, estimatedBytes);
  }

  /**
   * Add files from file input or drag-and-drop
   */
  async addFiles(fileList: FileList | File[]): Promise<void> {
    const files = Array.from(fileList);
    if (!files.length) return;

    const newItems: ImageItem[] = [];

    for (const file of files) {
      const ext = file.name.split('.').pop()?.toLowerCase() || 'unknown';
      const id = `${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;

      // Create preview URL
      let previewUrl = '';
      try {
        previewUrl = URL.createObjectURL(file);
      } catch {
        previewUrl = '';
      }

      newItems.push({
        id,
        file,
        name: file.name,
        originalFormat: ext.toUpperCase(),
        originalSize: file.size,
        previewUrl,
        rotation: 0,
        status: 'idle',
        progress: 0,
      });
    }

    this.images.update((current) => [...current, ...newItems]);
    this.step.set('workspace');
  }

  /**
   * Remove an image from the queue
   */
  removeImage(id: string): void {
    const item = this.images().find((i) => i.id === id);
    if (item?.previewUrl) {
      URL.revokeObjectURL(item.previewUrl);
    }
    if (item?.convertedUrl) {
      URL.revokeObjectURL(item.convertedUrl);
    }

    this.images.update((current) => current.filter((img) => img.id !== id));

    if (this.images().length === 0) {
      this.step.set('landing');
    }
  }

  /**
   * Rotate image 90 degrees clockwise
   */
  rotateImage(id: string): void {
    this.images.update((current) =>
      current.map((img) => {
        if (img.id === id) {
          const nextRotation = (img.rotation + 90) % 360;
          return { ...img, rotation: nextRotation };
        }
        return img;
      })
    );
  }

  setTargetFormat(format: SupportedFormat): void {
    this.targetFormat.set(format);
  }

  setQuality(quality: number): void {
    this.quality.set(quality);
  }

  setPreserveTransparency(preserve: boolean): void {
    this.preserveTransparency.set(preserve);
  }

  setIcoSize(size: number): void {
    this.icoSize.set(size);
  }

  /**
   * Start converting all queued images
   */
  async startConversion(): Promise<void> {
    const items = this.images();
    if (!items.length || this.isConverting()) return;

    this.isConverting.set(true);
    this.overallProgress.set(0);

    const settings: ConvertSettings = {
      targetFormat: this.targetFormat(),
      quality: this.quality(),
      preserveTransparency: this.preserveTransparency(),
      icoSize: this.icoSize(),
    };

    const total = items.length;

    for (let index = 0; index < total; index++) {
      this.currentConvertingIndex.set(index);
      const item = items[index];

      // Mark converting
      this.updateItem(item.id, { status: 'converting', progress: 20 });

      try {
        const result = await this.converter.convertSingleItem(item, settings);
        this.updateItem(item.id, {
          status: 'done',
          progress: 100,
          convertedBlob: result.blob,
          convertedUrl: result.url,
          convertedName: result.newName,
          convertedSize: result.size,
        });
      } catch (err: any) {
        console.error(`Error converting ${item.name}:`, err);
        this.updateItem(item.id, {
          status: 'error',
          progress: 0,
          errorMessage: err.message || 'Conversion failed',
        });
      }

      this.overallProgress.set(Math.round(((index + 1) / total) * 100));
    }

    this.isConverting.set(false);
    this.step.set('result');
  }

  private updateItem(id: string, patch: Partial<ImageItem>): void {
    this.images.update((current) =>
      current.map((item) => (item.id === id ? { ...item, ...patch } : item))
    );
  }

  /**
   * Download all converted files (ZIP if multiple, direct file if single)
   */
  async downloadAll(): Promise<void> {
    const doneItems = this.images().filter((img) => img.status === 'done' && img.convertedBlob);
    if (!doneItems.length) return;

    if (doneItems.length === 1) {
      const item = doneItems[0];
      this.converter.downloadBlob(item.convertedBlob!, item.convertedName!);
      return;
    }

    // Multiple files -> JSZip
    const zipBlob = await this.converter.createZipArchive(doneItems);
    const target = this.targetFormat().toUpperCase();
    this.converter.downloadBlob(zipBlob, `converted_images_${target}.zip`);
  }

  /**
   * Download a single image item
  */
  downloadSingle(id: string): void {
    const item = this.images().find((img) => img.id === id);
    if (item?.convertedBlob && item?.convertedName) {
      this.converter.downloadBlob(item.convertedBlob, item.convertedName);
    }
  }

  /**
   * Return to workspace
   */
  goToWorkspace(): void {
    this.step.set('workspace');
  }

  /**
   * Reset everything and return to landing
   */
  reset(): void {
    this.images().forEach((img) => {
      if (img.previewUrl) URL.revokeObjectURL(img.previewUrl);
      if (img.convertedUrl) URL.revokeObjectURL(img.convertedUrl);
    });
    this.images.set([]);
    this.overallProgress.set(0);
    this.isConverting.set(false);
    this.step.set('landing');
  }
}
