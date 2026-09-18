import { CommonModule } from '@angular/common';
import { Component, ElementRef, ViewChild, inject } from '@angular/core';
import { ConversionStateService } from '../../services/conversion-state.service';
import { ImageItem, SupportedFormat } from '../../models/image-item.model';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-workspace',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="workspace-3d-wrapper">
      <!-- Main 3D Image Queue Area -->
      <main class="queue-3d-area">
        <div class="queue-3d-header">
          <div class="queue-title-group">
            <h2 class="queue-3d-title">Conversion Queue</h2>
            <span class="queue-3d-count-pill">{{ state.totalImagesCount() }} {{ state.totalImagesCount() === 1 ? 'file' : 'files' }}</span>
          </div>

          <button
            type="button"
            class="btn-3d-secondary btn-add-more-3d"
            (click)="triggerFileInput()"
            title="Add more files"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            <span>Add More Files</span>
          </button>
        </div>

        <!-- Multi-image 3D Cards Grid (Dynamic Responsive for all screens) -->
        <div class="cards-3d-grid">
          @for (item of state.images(); track item.id) {
            <div class="image-3d-card">
              <!-- Floating 3D Actions: Rotate & Delete -->
              <div class="card-3d-actions">
                <button
                  type="button"
                  class="action-btn-3d"
                  title="Rotate 90° Clockwise"
                  (click)="state.rotateImage(item.id)"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
                  </svg>
                </button>
                <button
                  type="button"
                  class="action-btn-3d delete-btn"
                  title="Remove Image"
                  (click)="state.removeImage(item.id)"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                  </svg>
                </button>
              </div>

              <!-- 3D Preview Stage -->
              <div class="card-3d-preview-stage">
                @if (item.previewUrl) {
                  <img
                    [src]="item.previewUrl"
                    [alt]="item.name"
                    [style.transform]="'rotate(' + item.rotation + 'deg)'"
                    class="thumb-3d-img"
                  />
                } @else {
                  <div class="thumb-3d-fallback">
                    <span>{{ item.originalFormat }}</span>
                  </div>
                }

                <!-- 3D Holographic Conversion Laser Scanner -->
                @if (item.status === 'converting') {
                  <div class="hologram-scan-overlay">
                    <div class="laser-beam"></div>
                    <div class="scan-particles">
                      <div class="spinner-3d"></div>
                    </div>
                  </div>
                }
              </div>

              <!-- 3D Card Info Footer -->
              <div class="card-3d-details">
                <p class="file-name" [title]="item.name">{{ item.name }}</p>
                <div class="file-meta-row">
                  <span class="badge-format-3d">{{ item.originalFormat }}</span>
                  <span class="meta-size">{{ formatSize(item.originalSize) }}</span>
                  <span class="meta-arrow">➔</span>
                  <span class="meta-est-size">~{{ formatSize(state.getEstimatedSizeForItem(item)) }}</span>
                  @if (item.rotation > 0) {
                    <span class="meta-rot">{{ item.rotation }}°</span>
                  }
                </div>
              </div>
            </div>
          }
        </div>

        <!-- Hidden input for adding more files -->
        <input
          #fileInput
          type="file"
          multiple
          accept="image/*,.png,.jpg,.jpeg,.webp,.gif,.tif,.tiff,.psd,.svg,.bmp,.ico,.heic,.heif,.raw,.cr2,.nef,.arw,.dng,.orf,.pdf,.txt"
          style="display: none"
          (change)="onFilesSelected($event)"
        />
      </main>

      <!-- Right 3D Options Studio Panel (Sticky on Laptop/Windows Desktop) -->
      <aside class="sidebar-3d-panel">
        <div class="sidebar-3d-inner">
          <h3 class="panel-3d-heading">3D Convert Studio</h3>

          <!-- 1. Target Format Choice Chips -->
          <div class="setting-3d-block">
            <label class="setting-3d-label">Target Format</label>
            <div class="format-chips-3d-grid">
              @for (fmt of state.supportedFormats; track fmt.id) {
                <button
                  type="button"
                  class="chip-3d-btn"
                  [class.active]="state.targetFormat() === fmt.id"
                  (click)="state.setTargetFormat(fmt.id)"
                >
                  <span class="chip-3d-name">{{ fmt.id.toUpperCase() }}</span>
                  <span class="chip-3d-ext">.{{ fmt.ext }}</span>
                </button>
              }
            </div>
            <p class="format-3d-desc">{{ state.currentFormatOption().description }}</p>
          </div>

          <!-- 2. PRIMARY "CONVERT TO [FORMAT]" ACTION BUTTON (Placed directly below Target Format) -->
          <div class="top-convert-cta-block">
            @if (state.isConverting()) {
              <div class="progress-3d-box">
                <div class="progress-3d-track">
                  <div class="progress-3d-fill" [style.width.%]="state.overallProgress()"></div>
                </div>
                <p class="progress-3d-status">
                  Converting {{ state.currentConvertingIndex() + 1 }} of {{ state.totalImagesCount() }} ({{ state.overallProgress() }}%)
                </p>
              </div>
            }

            <button
              type="button"
              class="btn-3d-primary btn-convert-master-3d"
              [disabled]="state.isConverting() || state.totalImagesCount() === 0"
              (click)="state.startConversion()"
            >
              <span class="btn-text">
                Convert to {{ state.targetFormat().toUpperCase() }}
              </span>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <line x1="5" y1="12" x2="19" y2="12"></line>
                <polyline points="12 5 19 12 12 19"></polyline>
              </svg>
            </button>
          </div>

          <!-- 3. Quality Slider with Live Output Size -->
          @if (state.targetFormat() === 'jpg' || state.targetFormat() === 'webp') {
            <div class="setting-3d-block quality-3d-block">
              <div class="setting-label-row">
                <label class="setting-3d-label">Quality Level</label>
                <div class="quality-badges-group">
                  <span class="quality-3d-badge">{{ Math.round(state.quality() * 100) }}%</span>
                </div>
              </div>
              
              <div class="slider-3d-track-wrap">
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.01"
                  [value]="state.quality()"
                  (input)="onQualityChange($event)"
                  class="quality-3d-slider"
                />
              </div>
              <div class="slider-hint-row">
                <span>Smaller File</span>
                <span class="slider-3d-live-badge">
                  Live Est: <strong>~{{ formatSize(state.estimatedOutputSize()) }}</strong>
                </span>
                <span>Best Quality</span>
              </div>
            </div>
          }

          <!-- 4. Transparency preservation -->
          @if (['png', 'webp', 'svg', 'ico', 'tif', 'psd', 'pdf'].includes(state.targetFormat())) {
            <div class="setting-3d-block">
              <label class="toggle-3d-control">
                <input
                  type="checkbox"
                  [ngModel]="state.preserveTransparency()"
                  (ngModelChange)="state.setPreserveTransparency($event)"
                  class="checkbox-3d"
                />
                <span class="toggle-3d-text">Preserve Alpha Transparency</span>
              </label>
            </div>
          }

          <!-- 5. ICO Dimensions in 3D -->
          @if (state.targetFormat() === 'ico') {
            <div class="setting-3d-block">
              <label class="setting-3d-label">Icon Dimensions</label>
              <div class="ico-size-3d-row">
                <button
                  type="button"
                  class="size-pill-3d"
                  [class.active]="state.icoSize() === 0"
                  (click)="state.setIcoSize(0)"
                >
                  All / Multi-Res
                </button>
                @for (size of [512, 256, 128, 64, 32, 16]; track size) {
                  <button
                    type="button"
                    class="size-pill-3d"
                    [class.active]="state.icoSize() === size"
                    (click)="state.setIcoSize(size)"
                  >
                    {{ size }}px
                  </button>
                }
              </div>
            </div>
          }

          <!-- 6. 3D Queue Summary Podium -->
          <div class="queue-summary-3d-podium">
            <div class="summary-line">
              <span class="summary-key">Total Files:</span>
              <span class="summary-val">{{ state.totalImagesCount() }}</span>
            </div>
            <div class="summary-line">
              <span class="summary-key">Input Size:</span>
              <span class="summary-val">{{ formatSize(state.totalOriginalSize()) }}</span>
            </div>
            <div class="summary-line live-output-3d-line">
              <span class="summary-key">Est. Output Size:</span>
              <div class="live-est-group">
                <span class="summary-val output-size-3d-highlight">
                  ~{{ formatSize(state.estimatedOutputSize()) }}
                </span>
                @if (state.estimatedSavingsPercent() > 0) {
                  <span class="savings-pill-3d">-{{ state.estimatedSavingsPercent() }}%</span>
                }
              </div>
            </div>
            <div class="summary-line highlight-line">
              <span class="summary-key">Target Format:</span>
              <span class="summary-val target-badge-3d">{{ state.targetFormat().toUpperCase() }}</span>
            </div>
          </div>
        </div>
      </aside>

      <!-- Sticky Mobile Action Bar (Always accessible at bottom on mobile phones) -->
      <div class="mobile-sticky-action-bar">
        <div class="mobile-summary-pill">
          <span>{{ state.totalImagesCount() }} files</span>
          <span class="mobile-sep">•</span>
          <span class="mobile-est-size">~{{ formatSize(state.estimatedOutputSize()) }}</span>
        </div>
        <button
          type="button"
          class="btn-3d-primary btn-mobile-convert"
          [disabled]="state.isConverting() || state.totalImagesCount() === 0"
          (click)="state.startConversion()"
        >
          <span>Convert to {{ state.targetFormat().toUpperCase() }}</span>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <line x1="5" y1="12" x2="19" y2="12"></line>
            <polyline points="12 5 19 12 12 19"></polyline>
          </svg>
        </button>
      </div>
    </div>
  `,
  styles: [`
    .workspace-3d-wrapper {
      display: flex;
      min-height: calc(100vh - 68px);
      background-color: var(--bg-page);
      perspective: 1200px;
      transition: background-color 0.3s ease;
      position: relative;
    }

    /* Queue Main Content Area */
    .queue-3d-area {
      flex: 1;
      padding: 30px 34px;
      min-width: 0;
    }

    .queue-3d-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 14px;
    }

    .queue-title-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .queue-3d-title {
      font-size: clamp(20px, 2.5vw, 26px);
      font-weight: 900;
      color: var(--text-main);
      letter-spacing: -0.5px;
    }

    .queue-3d-count-pill {
      font-size: 12px;
      font-weight: 800;
      background: var(--bg-card);
      color: var(--primary);
      padding: 4px 12px;
      border-radius: var(--radius-pill);
      border: 1px solid var(--border-subtle);
      box-shadow: 0 2px 0 var(--border-hover);
    }

    .btn-add-more-3d {
      padding: 10px 18px;
      font-size: 13px;
      gap: 8px;
    }

    /* Dynamic Multi-column responsive grid */
    .cards-3d-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 20px;
    }

    /* 3D Image Card */
    .image-3d-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      overflow: hidden;
      box-shadow: var(--3d-card-shadow);
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      display: flex;
      flex-direction: column;
      position: relative;
      transform: translateY(0) rotateX(0deg);
      transform-style: preserve-3d;
    }

    .image-3d-card:hover {
      box-shadow: var(--3d-card-shadow-hover);
      transform: translateY(-4px) rotateX(2deg);
      border-color: var(--border-hover);
    }

    .card-3d-actions {
      position: absolute;
      top: 10px;
      right: 10px;
      display: flex;
      gap: 7px;
      z-index: 10;
      transform: translateZ(20px);
    }

    .action-btn-3d {
      width: 30px;
      height: 30px;
      border-radius: 50%;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-body);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 3px 0 var(--border-hover), 0 4px 8px rgba(0, 0, 0, 0.08);
      transform: translateY(0);
      transition: all 0.12s ease;
    }

    .action-btn-3d:hover {
      background: var(--primary);
      color: #ffffff;
      border-color: var(--primary);
      transform: translateY(-2px) scale(1.05);
      box-shadow: 0 4px 0 var(--3d-btn-primary-edge);
    }

    .action-btn-3d.delete-btn:hover {
      background: var(--danger);
      color: #ffffff;
      border-color: var(--danger);
      box-shadow: 0 4px 0 var(--danger-hover);
    }

    .action-btn-3d:active {
      transform: translateY(2px);
      box-shadow: 0 0 0 var(--border-hover);
    }

    .card-3d-preview-stage {
      height: 175px;
      background-color: var(--bg-subtle);
      background-image: 
        linear-gradient(45deg, var(--checker-color) 25%, transparent 25%),
        linear-gradient(-45deg, var(--checker-color) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, var(--checker-color) 75%),
        linear-gradient(-45deg, transparent 75%, var(--checker-color) 75%);
      background-size: 16px 16px;
      background-position: 0 0, 0 8px, 8px -8px, -8px 0px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 14px;
      position: relative;
      overflow: hidden;
    }

    .card-cutout-badge {
      position: absolute;
      top: 10px;
      left: 10px;
      background: var(--primary-gradient);
      color: #ffffff;
      font-size: 10px;
      font-weight: 800;
      padding: 3px 8px;
      border-radius: var(--radius-pill);
      box-shadow: 0 2px 0 var(--3d-btn-primary-edge), 0 4px 10px rgba(37, 99, 235, 0.4);
      z-index: 10;
      letter-spacing: 0.3px;
      animation: pulseLive 3s infinite ease-in-out;
    }

    .thumb-3d-img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border-radius: var(--radius-sm);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
      transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .thumb-3d-fallback {
      font-size: 20px;
      font-weight: 900;
      color: var(--text-light);
    }

    /* 3D Hologram Scanner */
    .hologram-scan-overlay {
      position: absolute;
      inset: 0;
      background: rgba(15, 23, 42, 0.7);
      backdrop-filter: blur(3px);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    .laser-beam {
      position: absolute;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, transparent, #38bdf8, #60a5fa, #38bdf8, transparent);
      box-shadow: 0 0 16px #38bdf8, 0 0 32px #3b82f6;
      animation: hologramScan 1.6s infinite ease-in-out;
    }

    .spinner-3d {
      width: 32px;
      height: 32px;
      border: 3px solid rgba(56, 189, 248, 0.3);
      border-top-color: #38bdf8;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    .card-3d-details {
      padding: 12px 14px;
      border-top: 1px solid var(--border-subtle);
      background: var(--bg-card);
    }

    .file-name {
      font-size: 13px;
      font-weight: 700;
      color: var(--text-main);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-bottom: 6px;
    }

    .file-meta-row {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      color: var(--text-muted);
      flex-wrap: wrap;
    }

    .badge-format-3d {
      background: var(--bg-subtle);
      color: var(--text-body);
      padding: 2px 7px;
      border-radius: 4px;
      font-weight: 800;
      font-size: 10px;
      border: 1px solid var(--border-subtle);
      box-shadow: 0 1px 0 var(--border-hover);
    }

    .meta-est-size {
      color: var(--primary);
      font-weight: 800;
    }

    .meta-rot {
      color: var(--primary);
      font-weight: 800;
      margin-left: auto;
    }

    /* 3D Sidebar Studio Panel (Sticky on Desktop/Windows PC) */
    .sidebar-3d-panel {
      width: 360px;
      min-width: 340px;
      background: var(--bg-card);
      border-left: 1px solid var(--border-subtle);
      position: sticky;
      top: 68px;
      height: calc(100vh - 68px);
      overflow-y: auto;
      box-shadow: -4px 0 20px rgba(0, 0, 0, 0.03);
      transition: background-color 0.3s ease, border-color 0.3s ease;
      z-index: 20;
    }

    .sidebar-3d-inner {
      padding: 28px 24px 40px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .panel-3d-heading {
      font-size: 20px;
      font-weight: 900;
      color: var(--text-main);
      letter-spacing: -0.4px;
      margin-bottom: 4px;
    }

    .setting-3d-block {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .setting-3d-label {
      display: block;
      font-size: 12px;
      font-weight: 800;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }

    .setting-label-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .quality-3d-badge {
      font-size: 13px;
      font-weight: 800;
      color: var(--primary);
      background: var(--primary-light);
      padding: 2px 10px;
      border-radius: var(--radius-pill);
      border: 1px solid var(--border-subtle);
      box-shadow: 0 1px 0 var(--border-hover);
    }

    /* 3D Format Chips */
    .format-chips-3d-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }

    .chip-3d-btn {
      background: var(--bg-subtle);
      border: 1px solid var(--border-subtle);
      color: var(--text-body);
      padding: 8px 6px;
      border-radius: var(--radius-sm);
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 2px;
      box-shadow: 0 3px 0 var(--border-hover);
      transform: translateY(0);
      transition: all 0.12s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .chip-3d-btn:hover {
      background: var(--bg-card-hover);
      transform: translateY(-1px);
      box-shadow: 0 4px 0 var(--border-hover);
      color: var(--text-main);
    }

    .chip-3d-btn:active {
      transform: translateY(2px);
      box-shadow: 0 0 0 var(--border-hover);
    }

    .chip-3d-btn.active {
      background: var(--primary-gradient);
      border-color: var(--primary);
      color: #ffffff;
      box-shadow: 0 3px 0 var(--3d-btn-primary-edge), 0 6px 14px rgba(37, 99, 235, 0.35);
    }

    .chip-3d-name {
      font-size: 13px;
      font-weight: 900;
    }

    .chip-3d-ext {
      font-size: 10px;
      opacity: 0.85;
    }

    .format-3d-desc {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.4;
    }

    /* Top Convert CTA Block (Right below Target Format) */
    .top-convert-cta-block {
      background: var(--bg-subtle);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      box-shadow: 0 3px 0 var(--border-subtle);
    }

    .btn-convert-master-3d {
      width: 100%;
      font-size: 16px;
      padding: 16px 20px;
      gap: 10px;
    }

    .progress-3d-box {
      margin-bottom: 4px;
    }

    .progress-3d-track {
      width: 100%;
      height: 8px;
      background: var(--border-subtle);
      border-radius: 4px;
      overflow: hidden;
      box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
      margin-bottom: 6px;
    }

    .progress-3d-fill {
      height: 100%;
      background: var(--primary-gradient);
      box-shadow: 0 0 10px var(--primary-glow);
      transition: width 0.3s ease;
    }

    .progress-3d-status {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-muted);
      text-align: center;
    }

    /* 3D Slider */
    .quality-3d-slider {
      width: 100%;
      height: 8px;
      border-radius: 4px;
      background: var(--border-hover);
      outline: none;
      appearance: none;
      box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.15);
      cursor: pointer;
    }

    .quality-3d-slider::-webkit-slider-thumb {
      appearance: none;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: radial-gradient(circle at 35% 35%, #60a5fa, #1d4ed8);
      cursor: pointer;
      box-shadow: 0 4px 8px rgba(37, 99, 235, 0.5), inset 0 2px 0 rgba(255, 255, 255, 0.4);
      border: 2px solid #ffffff;
      transition: transform 0.1s ease;
    }

    .quality-3d-slider::-webkit-slider-thumb:hover,
    .quality-3d-slider::-webkit-slider-thumb:active {
      transform: scale(1.2);
    }

    .slider-hint-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
      color: var(--text-light);
      margin-top: 6px;
    }

    .slider-3d-live-badge {
      font-size: 11px;
      color: var(--primary);
      background: var(--primary-light);
      padding: 2px 8px;
      border-radius: var(--radius-xs);
      border: 1px solid var(--border-subtle);
      box-shadow: 0 1px 0 var(--border-hover);
    }

    .slider-3d-live-badge strong {
      color: var(--primary);
      font-weight: 800;
    }

    .toggle-3d-control {
      display: flex;
      align-items: center;
      gap: 10px;
      cursor: pointer;
    }

    .checkbox-3d {
      width: 18px;
      height: 18px;
      accent-color: var(--primary);
      cursor: pointer;
      flex-shrink: 0;
    }

    .toggle-3d-text {
      font-size: 13px;
      font-weight: 800;
      color: var(--text-main);
    }

    .ico-size-3d-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 6px;
    }

    .size-pill-3d {
      padding: 7px 0;
      background: var(--bg-subtle);
      color: var(--text-body);
      border-radius: var(--radius-sm);
      font-size: 11px;
      font-weight: 800;
      border: 1px solid var(--border-subtle);
      box-shadow: 0 2px 0 var(--border-hover);
    }

    .size-pill-3d.active {
      background: var(--primary-gradient);
      color: #ffffff;
      border-color: var(--primary);
      box-shadow: 0 2px 0 var(--3d-btn-primary-edge);
    }

    /* 3D Queue Summary Podium */
    .queue-summary-3d-podium {
      background: var(--bg-subtle);
      border-radius: var(--radius-md);
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      border: 1px solid var(--border-subtle);
      box-shadow: 0 3px 0 var(--border-subtle);
    }

    .summary-line {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
      color: var(--text-muted);
    }

    .summary-val {
      font-weight: 800;
      color: var(--text-main);
    }

    .live-output-3d-line {
      background: var(--bg-card);
      padding: 8px 12px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-focus);
      box-shadow: 0 2px 0 var(--border-hover);
      margin: 2px 0;
    }

    .live-output-3d-line .summary-key {
      font-weight: 800;
      color: var(--text-main);
    }

    .output-size-3d-highlight {
      font-size: 14px;
      font-weight: 900;
      color: var(--primary);
    }

    .savings-pill-3d {
      background: var(--success-bg);
      color: var(--success-text);
      border: 1px solid var(--success-border);
      font-size: 11px;
      font-weight: 900;
      padding: 2px 7px;
      border-radius: var(--radius-pill);
    }

    .target-badge-3d {
      color: var(--primary);
      font-weight: 900;
    }

    /* Mobile Sticky Bottom Bar (Only visible on small mobile screens) */
    .mobile-sticky-action-bar {
      display: none;
    }

    /* ==========================================================================
       DYNAMIC RESPONSIVENESS: LAPTOPS, TABLETS, MOBILES
       ========================================================================== */

    /* Tablet & Medium Laptops (max-width: 1024px) */
    @media (max-width: 1024px) {
      .workspace-3d-wrapper {
        flex-direction: column;
      }
      .sidebar-3d-panel {
        width: 100%;
        min-width: 100%;
        position: static;
        height: auto;
        border-left: none;
        border-top: 1px solid var(--border-subtle);
      }
      .sidebar-3d-inner {
        padding: 24px 20px;
      }
      .queue-3d-area {
        padding: 24px 20px;
      }
      .cards-3d-grid {
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 16px;
      }
    }

    /* Mobile Phones (max-width: 640px) */
    @media (max-width: 640px) {
      .queue-3d-area {
        padding: 16px 14px 90px; /* Extra bottom padding for mobile sticky bar */
      }
      .cards-3d-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
      }
      .card-3d-preview-stage {
        height: 140px;
        padding: 10px;
      }
      .card-3d-details {
        padding: 10px 12px;
      }
      .file-name {
        font-size: 12px;
      }
      .file-meta-row {
        font-size: 10px;
      }
      .btn-add-more-3d {
        padding: 8px 14px;
        font-size: 12px;
      }

      /* Enable mobile sticky action bar */
      .mobile-sticky-action-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: var(--bg-glass);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-top: 1px solid var(--border-subtle);
        padding: 12px 16px;
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.15);
        z-index: 100;
        gap: 12px;
      }

      .mobile-summary-pill {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 700;
        color: var(--text-main);
      }

      .mobile-sep {
        color: var(--text-light);
      }

      .mobile-est-size {
        color: var(--primary);
        font-weight: 800;
      }

      .btn-mobile-convert {
        flex: 1;
        font-size: 14px;
        padding: 12px 16px;
        gap: 6px;
      }
    }

    /* Small Mobile Phones (max-width: 380px) */
    @media (max-width: 380px) {
      .cards-3d-grid {
        grid-template-columns: 1fr;
      }
    }
  `]
})
export class WorkspaceComponent {
  state = inject(ConversionStateService);
  Math = Math;

  @ViewChild('fileInput') fileInputRef!: ElementRef<HTMLInputElement>;

  triggerFileInput(): void {
    if (this.fileInputRef) {
      this.fileInputRef.nativeElement.value = '';
      this.fileInputRef.nativeElement.click();
    }
  }

  onFilesSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.state.addFiles(input.files);
    }
  }

  onQualityChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.state.setQuality(parseFloat(input.value));
  }

  formatSize(bytes: number): string {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }
}
