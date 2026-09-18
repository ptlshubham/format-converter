export type SupportedFormat =
  | 'webp'
  | 'jpg'
  | 'png'
  | 'pdf'
  | 'gif'
  | 'tif'
  | 'psd'
  | 'svg'
  | 'bmp'
  | 'ico'
  | 'heic'
  | 'raw';

export interface FormatOption {
  id: SupportedFormat;
  label: string;
  ext: string;
  mimeType: string;
  description: string;
}

export interface ImageItem {
  id: string;
  file: File;
  name: string;
  originalFormat: string;
  originalSize: number;
  previewUrl: string;
  width?: number;
  height?: number;
  rotation: number; // 0, 90, 180, 270
  status: 'idle' | 'converting' | 'done' | 'error';
  progress: number;
  convertedBlob?: Blob;
  convertedUrl?: string;
  convertedName?: string;
  convertedSize?: number;
  errorMessage?: string;
}

export interface ConvertSettings {
  targetFormat: SupportedFormat;
  quality: number; // 0.1 - 1.0
  preserveTransparency: boolean;
  icoSize: number;
}
