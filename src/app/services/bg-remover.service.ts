import { Injectable, computed, signal } from '@angular/core';
import { environment } from '../../environments/environment';

export interface BgRemoveResponse {
  success: boolean;
  requestId?: string;
  engine?: string;
  model?: string;
  device?: string;
  width: number;
  height: number;
  processingTimeMs: number;
  originalFormat: string;
  resultDataUri: string;
  privacyNotice: string;
  diagnostics?: any;
}

export type BgType = 'transparent' | 'solid' | 'gradient' | 'image';
export type ExportQuality = 'original' | 'hd2x' | 'hd4x';

@Injectable({
  providedIn: 'root',
})
export class BgRemoverService {
  private readonly apiUrl = environment.apiUrl;


  // State Signals
  readonly originalFile = signal<File | null>(null);
  readonly originalPreviewUrl = signal<string | null>(null);
  readonly originalDimensions = signal<{ width: number; height: number } | null>(null);

  readonly isProcessing = signal<boolean>(false);
  readonly isExporting = signal<boolean>(false);
  readonly errorMessage = signal<string | null>(null);

  readonly processedResultUri = signal<string | null>(null);
  readonly processingTime = signal<number | null>(null);
  readonly lastRequestId = signal<string | null>(null);
  readonly lastEngine = signal<string | null>(null);
  readonly lastModel = signal<string | null>(null);
  readonly lastDevice = signal<string | null>(null);
  readonly lastProvider = signal<string | null>('CPU');
  readonly lastOutputSizeStr = signal<string | null>(null);

  readonly exportTimeMs = signal<number | null>(null);
  readonly exportTimeSec = computed<string>(() => {
    const ms = this.exportTimeMs();
    return ms !== null ? (ms / 1000).toFixed(3) : '0.000';
  });
  readonly processingTimeSec = computed<string>(() => {
    const ms = this.processingTime();
    return ms !== null ? (ms / 1000).toFixed(3) : '0.000';
  });

  // Customization controls (0-100 percentage values)
  readonly sensitivity = signal<number>(10);
  readonly edgeSoftness = signal<number>(50);
  readonly defringeStrength = signal<number>(50);

  // Background Customizer settings
  readonly bgType = signal<BgType>('transparent');
  readonly solidColor = signal<string>('#ffffff');
  readonly gradientColor1 = signal<string>('#6366f1');
  readonly gradientColor2 = signal<string>('#ec4899');
  readonly gradientDirection = signal<string>('to-bottom');
  readonly customBgImageUrl = signal<string | null>(null);

  // Split comparison slider position (0 to 100%)
  readonly splitPosition = signal<number>(50);

  // Phase 2: Export Quality state ('original' | 'hd2x' | 'hd4x')
  readonly exportQuality = signal<ExportQuality>('original');

  // Computed target output dimensions based on selected export quality
  readonly targetExportDimensions = computed<{ width: number; height: number; multiplier: number; label: string } | null>(() => {
    const dims = this.originalDimensions();
    if (!dims) return null;
    const quality = this.exportQuality();
    const mult = quality === 'hd4x' ? 4 : quality === 'hd2x' ? 2 : 1;
    const label = quality === 'hd4x' ? 'Ultra HD 4×' : quality === 'hd2x' ? 'HD 2×' : 'Native';
    return {
      width: dims.width * mult,
      height: dims.height * mult,
      multiplier: mult,
      label,
    };
  });

  // Default values for Custom Algorithm Tuning
  readonly defaultSensitivity = 10;
  readonly defaultEdgeSoftness = 50;
  readonly defaultDefringeStrength = 50;

  readonly hasTuningChanged = computed<boolean>(() => {
    return (
      this.sensitivity() !== this.defaultSensitivity ||
      this.edgeSoftness() !== this.defaultEdgeSoftness ||
      this.defringeStrength() !== this.defaultDefringeStrength
    );
  });

  resetTuning(): void {
    this.sensitivity.set(this.defaultSensitivity);
    this.edgeSoftness.set(this.defaultEdgeSoftness);
    this.defringeStrength.set(this.defaultDefringeStrength);
  }

  setImage(file: File): void {
    this.originalFile.set(file);
    this.errorMessage.set(null);
    this.processedResultUri.set(null);
    this.processingTime.set(null);
    this.lastRequestId.set(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      const url = e.target?.result as string;
      this.originalPreviewUrl.set(url);

      const img = new Image();
      img.onload = () => {
        this.originalDimensions.set({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.src = url;

      // User workflow: User uploads image, inspects or tweaks settings, then clicks Proceed
    };
    reader.readAsDataURL(file);
  }

  setImageFromBlob(blob: Blob, fileName: string): void {
    const file = new File([blob], fileName, { type: blob.type || 'image/png' });
    this.setImage(file);
  }

  async processBackgroundRemoval(): Promise<void> {
    // Rule 6: Guard against duplicate / concurrent frontend requests
    if (this.isProcessing()) {
      console.warn('Background removal inference is already in progress. Ignoring duplicate request.');
      return;
    }

    const file = this.originalFile();
    if (!file) return;

    this.isProcessing.set(true);
    this.errorMessage.set(null);
    // Never show stale result while processing
    this.processedResultUri.set(null);

    const formData = new FormData();
    formData.append('file', file);
    // Transmit integer percentage 0-100 (e.g., 10 for 10%, 50 for 50%)
    formData.append('sensitivity', Math.round(this.sensitivity()).toString());
    formData.append('edge_softness', Math.round(this.edgeSoftness()).toString());
    formData.append('defringe_strength', Math.round(this.defringeStrength()).toString());

    try {
      const response = await fetch(`${this.apiUrl}/remove`, {
        method: 'POST',
        body: formData,
        cache: 'no-store',
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({ detail: 'Server error' }));
        if (response.status === 429) {
          throw new Error(errJson.detail || 'Background removal is currently busy with another image. Please wait a moment.');
        }
        throw new Error(errJson.detail || `HTTP error ${response.status}`);
      }

      const data: BgRemoveResponse = await response.json();
      if (data.success && data.resultDataUri) {
        this.processedResultUri.set(data.resultDataUri);
        this.processingTime.set(data.processingTimeMs);
        if (data.requestId) this.lastRequestId.set(data.requestId);
        if (data.engine) this.lastEngine.set(data.engine);
        if (data.model) this.lastModel.set(data.model);
      } else {
        throw new Error('Invalid response from Background Remover engine.');
      }
    } catch (err: any) {
      console.error('BG Removal Error:', err);
      this.errorMessage.set(
        err.message || 'Failed to connect to Background Remover service. Ensure backend is running.'
      );
    } finally {
      this.isProcessing.set(false);
    }
  }

  setCustomBgImage(file: File): void {
    const reader = new FileReader();
    reader.onload = (e) => {
      this.customBgImageUrl.set(e.target?.result as string);
      this.bgType.set('image');
    };
    reader.readAsDataURL(file);
  }

  reset(): void {
    this.originalFile.set(null);
    this.originalPreviewUrl.set(null);
    this.originalDimensions.set(null);
    this.processedResultUri.set(null);
    this.processingTime.set(null);
    this.errorMessage.set(null);
    this.customBgImageUrl.set(null);
    this.bgType.set('transparent');
    this.exportQuality.set('original');
    this.isExporting.set(false);
  }

  /**
   * Generates a composite canvas and triggers download.
   * PHASE 1: For Original Quality + Transparent background PNG, downloads the raw backend PNG
   * directly without canvas re-encoding to preserve 100% native resolution, true RGBA alpha,
   * and zero lossy compression artifacts.
   * PHASE 2: Supports HD 2x and Ultra HD 4x High-Quality Upscaling using high-precision canvas smoothing.
   */
  async downloadResult(format: 'png' | 'webp' | 'jpg' = 'png', quality: number = 0.95): Promise<void> {
    const fgUri = this.processedResultUri();
    if (!fgUri || this.isExporting()) return;

    this.isExporting.set(true);

    try {
      const baseName = this.originalFile()?.name.replace(/\.[^/.]+$/, '') || 'cutout';
      const type = this.bgType();
      const expQuality = this.exportQuality();
      const scaleMult = expQuality === 'hd4x' ? 4 : expQuality === 'hd2x' ? 2 : 1;

      // REQUIREMENT: AI Super Resolution Export via Backend (Real-ESRGAN x4plus)
      if (expQuality === 'hd2x' || expQuality === 'hd4x') {
        const payload = {
          image_base64: fgUri,
          export_quality: expQuality,
          export_format: format.toUpperCase(),
          bg_type: type,
          color1: this.solidColor(),
          color2: this.gradientColor2(),
          gradient_direction: this.gradientDirection(),
          quality: Math.round(quality * 100),
        };

        const response = await fetch(`${this.apiUrl}/export-hd`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const errJson = await response.json().catch(() => ({ detail: 'Export failed' }));
          throw new Error(errJson.detail || 'AI Super Resolution export failed.');
        }

        const resData = await response.json();
        if (resData.success && resData.resultDataUri) {
          if (resData.perfMetrics) {
            this.exportTimeMs.set(resData.perfMetrics.total_export_ms || resData.perfMetrics.totalMs || null);
            if (resData.perfMetrics.provider) this.lastProvider.set(resData.perfMetrics.provider);
          }
          const suffix = expQuality === 'hd4x' ? '-4k-ai' : '-hd-ai';
          const link = document.createElement('a');
          link.download = `${baseName}${suffix}.${format}`;
          link.href = resData.resultDataUri;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          return;
        }
      }

      // REQUIREMENT: Original Quality + PNG + Transparent background -> Direct Download (No canvas re-encoding)
      if (format === 'png' && type === 'transparent' && expQuality === 'original') {
        const link = document.createElement('a');
        link.download = `${baseName}-nobg.png`;
        link.href = fgUri; // Direct raw backend lossless PNG Data URI
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        return;
      }

      // REQUIREMENT: For HD 2x/4x or WebP/JPG or custom background PNG: Render via high-precision canvas upscaling
      const fgImg = await this.loadImage(fgUri);
      const targetW = fgImg.naturalWidth * scaleMult;
      const targetH = fgImg.naturalHeight * scaleMult;

      // Safe Canvas Dimension Bounds Check (Max 16,384px per side to prevent browser canvas crashes)
      const MAX_CANVAS_DIM = 16384;
      const finalW = Math.min(targetW, MAX_CANVAS_DIM);
      const finalH = Math.min(targetH, MAX_CANVAS_DIM);

      const canvas = document.createElement('canvas');
      canvas.width = finalW;
      canvas.height = finalH;
      const ctx = canvas.getContext('2d', { alpha: format !== 'jpg' });
      if (!ctx) return;

      // High Quality Image Smoothing for High-Quality Upscaling
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';

      // 1. Render Background Layer
      if (type === 'solid' || (format === 'jpg' && type === 'transparent')) {
        ctx.fillStyle = type === 'solid' ? this.solidColor() : '#ffffff';
        ctx.fillRect(0, 0, finalW, finalH);
      } else if (type === 'gradient') {
        let grad: CanvasGradient;
        const dir = this.gradientDirection();
        if (dir === 'to-right') {
          grad = ctx.createLinearGradient(0, 0, finalW, 0);
        } else if (dir === 'diagonal') {
          grad = ctx.createLinearGradient(0, 0, finalW, finalH);
        } else {
          grad = ctx.createLinearGradient(0, 0, 0, finalH);
        }
        grad.addColorStop(0, this.gradientColor1());
        grad.addColorStop(1, this.gradientColor2());
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, finalW, finalH);
      } else if (type === 'image' && this.customBgImageUrl()) {
        const bgImg = await this.loadImage(this.customBgImageUrl()!);
        ctx.drawImage(bgImg, 0, 0, finalW, finalH);
      }

      // 2. Render Foreground Cutout (scaled with high-quality smoothing)
      ctx.drawImage(fgImg, 0, 0, finalW, finalH);

      // 3. Export & Download with high quality (0.95+)
      const mimeType = format === 'jpg' ? 'image/jpeg' : format === 'webp' ? 'image/webp' : 'image/png';
      const exportQuality = format === 'png' ? 1.0 : quality;
      const dataUrl = canvas.toDataURL(mimeType, exportQuality);

      const suffix = expQuality === 'hd4x' ? '-4k' : expQuality === 'hd2x' ? '-hd' : '-nobg';
      const link = document.createElement('a');
      link.download = `${baseName}${suffix}.${format}`;
      link.href = dataUrl;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      console.error('Export failed:', err);
    } finally {
      this.isExporting.set(false);
    }
  }

  private loadImage(src: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }
}
