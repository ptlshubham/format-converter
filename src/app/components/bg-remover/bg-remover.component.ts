import { CommonModule } from '@angular/common';
import { Component, ElementRef, HostListener, ViewChild, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { BgRemoverService, BgType } from '../../services/bg-remover.service';

@Component({
  selector: 'app-bg-remover',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="bgr-shell">
      <!-- Top Privacy & Architecture Badge -->
      <div class="privacy-banner-card">
        <div class="privacy-badge">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
          </svg>
          <span>Privacy Guaranteed</span>
        </div>
        <p class="privacy-desc">
          Your image is processed purely in memory using our <strong>custom in-house computer vision engine</strong>.
          Uploaded images are <strong>never stored on the server</strong> and are immediately purged from memory after processing.
        </p>
      </div>

      <!-- State 1: Upload Dropzone -->
      @if (!bgService.originalPreviewUrl()) {
        <div class="bgr-upload-section">
          <div
            class="bgr-dropzone"
            [class.dragover]="isDragging"
            (dragover)="onDragOver($event)"
            (dragleave)="onDragLeave($event)"
            (drop)="onDrop($event)"
            (click)="fileInput.click()"
          >
            <input
              #fileInput
              type="file"
              accept="image/png,image/jpeg,image/webp,image/avif"
              style="display: none"
              (change)="onFileSelected($event)"
            />
            <div class="drop-icon-box">
              <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </div>
            <h2 class="drop-title">Drop an image here to Remove Background</h2>
            <p class="drop-subtitle">Supports JPG, PNG, WEBP, AVIF up to 25MB • 100% In-House CV Engine</p>
            <button type="button" class="btn-select-image">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <circle cx="8.5" cy="8.5" r="1.5"></circle>
                <polyline points="21 15 16 10 5 21"></polyline>
              </svg>
              Select Image
            </button>
          </div>
        </div>
      }

      <!-- State 2: Studio Workbench -->
      @if (bgService.originalPreviewUrl()) {
        <div class="bgr-workbench-layout">
          <!-- Left: Visual Canvas / Comparison Slider -->
          <div class="workbench-canvas-panel">
            <div class="canvas-header-bar">
              <div class="file-info-tag">
                <span class="file-name">{{ bgService.originalFile()?.name }}</span>
                @if (bgService.originalDimensions(); as dims) {
                  <span class="dims-badge">{{ dims.width }} × {{ dims.height }} px</span>
                }
                @if (bgService.lastRequestId(); as reqId) {
                  <span class="dims-badge" style="background: rgba(99, 102, 241, 0.2); color: #818cf8; font-weight: 600;">{{ reqId }}</span>
                }
              </div>

              @if (bgService.processingTime(); as time) {
                <div class="perf-pill">
                  <span class="dot-live"></span>
                  <span>Processed in {{ time }}ms</span>
                </div>
              }
            </div>

            <!-- Interactive Before/After Split Viewer -->
            <div
              class="comparison-container"
              #cmpContainer
              (mousedown)="onMouseDown($event)"
              (mousemove)="onMouseMove($event)"
              (mouseup)="onMouseUp()"
              (mouseleave)="onMouseLeave($event)"
              (touchstart)="onTouchStart($event)"
              (touchmove)="onTouchMove($event)"
              (touchend)="onTouchEnd()"
            >
              @if (bgService.isProcessing()) {
                <div class="processing-overlay">
                  <div class="scanner-laser"></div>
                  <div class="spinner-box">
                    <div class="cv-spinner"></div>
                    <span class="processing-label">Analyzing image contours & generating matte...</span>
                  </div>
                </div>
              }

              <!-- Error State -->
              @if (bgService.errorMessage(); as err) {
                <div class="error-banner">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                  </svg>
                  <span>{{ err }}</span>
                  <button class="btn-retry" [disabled]="bgService.isProcessing()" (click)="bgService.processBackgroundRemoval()">Retry</button>
                </div>
              }

              <!-- 1. Underlay Layer (Cutout Result over Checkerboard / Custom Background) -->
              <div
                class="canvas-bg-underlay"
                [ngClass]="bgService.bgType()"
                [style.background-color]="bgService.bgType() === 'solid' ? bgService.solidColor() : null"
                [style.background-image]="getUnderlayBackground()"
              >
                @if (bgService.processedResultUri(); as resUri) {
                  <img [src]="resUri" alt="Cutout Result" class="comparison-img cutout-img" />
                }
              </div>

              <!-- 2. Original Image Layer (Clipped using CSS clip-path inset for 100% pixel-perfect alignment) -->
              <div
                class="original-clip-layer"
                [style.clip-path]="bgService.processedResultUri() ? ('inset(0 ' + (100 - bgService.splitPosition()) + '% 0 0)') : 'none'"
              >
                <img [src]="bgService.originalPreviewUrl()" alt="Original" class="comparison-img original-img" />
              </div>

              <!-- 3. Labels & Handle (only when processed result exists) -->
              @if (bgService.processedResultUri()) {
                <span class="split-tag original-tag">ORIGINAL</span>
                <span class="split-tag result-tag">CUTOUT</span>

                <!-- 4. Split Handle Divider -->
                <div class="split-divider-handle" [style.left.%]="bgService.splitPosition()">
                  <div class="handle-pill">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                      <polyline points="15 18 9 12 15 6"></polyline>
                    </svg>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                      <polyline points="9 18 15 12 9 6"></polyline>
                    </svg>
                  </div>
                </div>
              } @else if (!bgService.isProcessing()) {
                <div class="unprocessed-guide-badge">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="16" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12.01" y2="8"></line>
                  </svg>
                  <span>Ready — Adjust algorithm tuning if desired, then click <strong>Proceed</strong></span>
                </div>
              }
            </div>

            @if (bgService.processedResultUri()) {
              <!-- Split slider range bar -->
              <div class="slider-control-row">
                <span class="slider-label">Comparison Split:</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  [value]="bgService.splitPosition()"
                  (input)="onSplitChange($event)"
                  class="custom-range"
                />
                <span class="slider-val">{{ bgService.splitPosition() }}%</span>
              </div>
            }
          </div>

          <!-- Right: Controls & Background Customizer Sidebar -->
          <div class="workbench-controls-panel">
            <!-- Action Toolbar -->
            <div class="action-top-row">
              <button class="btn-tool-secondary" (click)="bgService.reset()">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path>
                  <path d="M3 3v5h5"></path>
                </svg>
                Change Image
              </button>

              <button
                class="btn-reprocess"
                [disabled]="bgService.isProcessing()"
                (click)="bgService.processBackgroundRemoval()"
              >
                @if (bgService.isProcessing()) {
                  <span class="btn-spinner"></span>
                  Processing...
                } @else if (bgService.processedResultUri()) {
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"></path>
                  </svg>
                  Re-process
                } @else {
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                  </svg>
                  Proceed
                }
              </button>
            </div>

            <!-- Algorithm Fine-Tuning Controls -->
            <div class="control-card">
              <h3 class="card-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <line x1="4" y1="21" x2="4" y2="14"></line>
                  <line x1="4" y1="10" x2="4" y2="3"></line>
                  <line x1="12" y1="21" x2="12" y2="12"></line>
                  <line x1="12" y1="8" x2="12" y2="3"></line>
                  <line x1="20" y1="21" x2="20" y2="16"></line>
                  <line x1="20" y1="12" x2="20" y2="3"></line>
                </svg>
                Custom Algorithm Tuning
              </h3>

              <!-- Sensitivity -->
              <div class="slider-group">
                <div class="slider-header">
                  <label>Background Sensitivity</label>
                  <span>{{ bgService.sensitivity() }}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  [disabled]="bgService.isProcessing()"
                  [value]="bgService.sensitivity()"
                  (input)="onParamInput('sensitivity', $event)"
                  class="custom-range"
                />
                <span class="control-hint">Controls how aggressively uncertain peripheral background regions are rejected around the locked subject core.</span>
              </div>

              <!-- Edge Softness -->
              <div class="slider-group">
                <div class="slider-header">
                  <label>Edge Feathering / Softness</label>
                  <span>{{ bgService.edgeSoftness() }}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  [disabled]="bgService.isProcessing()"
                  [value]="bgService.edgeSoftness()"
                  (input)="onParamInput('edgeSoftness', $event)"
                  class="custom-range"
                />
                <span class="control-hint">Bilateral antialiasing strictly on the outer subject perimeter (subject core remains 100% solid).</span>
              </div>

              <!-- De-fringing -->
              <div class="slider-group">
                <div class="slider-header">
                  <label>De-Fringe / Halo Removal</label>
                  <span>{{ bgService.defringeStrength() }}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  [disabled]="bgService.isProcessing()"
                  [value]="bgService.defringeStrength()"
                  (input)="onParamInput('defringeStrength', $event)"
                  class="custom-range"
                />
                <span class="control-hint">Removes background color bleed strictly along semi-transparent boundary pixels without affecting the subject interior.</span>
              </div>

              <!-- Tuning Action Buttons: Reset and Proceed -->
              <div class="tuning-actions-row">
                <button
                  type="button"
                  class="btn-tuning-reset"
                  [disabled]="bgService.isProcessing()"
                  (click)="bgService.resetTuning()"
                  title="Reset all 3 tuning settings to default values"
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path>
                    <path d="M3 3v5h5"></path>
                  </svg>
                  Reset
                </button>

                <button
                  type="button"
                  class="btn-tuning-proceed"
                  [disabled]="bgService.isProcessing()"
                  (click)="bgService.processBackgroundRemoval()"
                >
                  @if (bgService.isProcessing()) {
                    <span class="btn-spinner"></span>
                    <span>Processing...</span>
                  } @else {
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                    <span>Proceed</span>
                  }
                </button>
              </div>
            </div>

            <!-- Background Replacement Studio -->
            <div class="control-card">
              <h3 class="card-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <circle cx="8.5" cy="8.5" r="1.5"></circle>
                  <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
                Replace Background
              </h3>

              <!-- Mode Tabs -->
              <div class="bg-mode-tabs">
                <button
                  type="button"
                  class="mode-tab"
                  [class.active]="bgService.bgType() === 'transparent'"
                  (click)="bgService.bgType.set('transparent')"
                >
                  Transparent
                </button>
                <button
                  type="button"
                  class="mode-tab"
                  [class.active]="bgService.bgType() === 'solid'"
                  (click)="bgService.bgType.set('solid')"
                >
                  Solid Color
                </button>
                <button
                  type="button"
                  class="mode-tab"
                  [class.active]="bgService.bgType() === 'gradient'"
                  (click)="bgService.bgType.set('gradient')"
                >
                  Gradient
                </button>
                <button
                  type="button"
                  class="mode-tab"
                  [class.active]="bgService.bgType() === 'image'"
                  (click)="bgFileInput.click()"
                >
                  Image
                </button>
                <input
                  #bgFileInput
                  type="file"
                  accept="image/*"
                  style="display: none"
                  (change)="onBgFileSelected($event)"
                />
              </div>

              <!-- Solid Color Palette -->
              @if (bgService.bgType() === 'solid') {
                <div class="solid-palette-box">
                  <div class="color-presets">
                    @for (c of presetColors; track c) {
                      <button
                        type="button"
                        class="color-chip"
                        [style.background-color]="c"
                        [class.selected]="bgService.solidColor() === c"
                        (click)="bgService.solidColor.set(c)"
                      ></button>
                    }
                  </div>
                  <div class="custom-color-picker">
                    <label>Custom:</label>
                    <input
                      type="color"
                      [value]="bgService.solidColor()"
                      (input)="onColorChange($event)"
                      class="color-input"
                    />
                    <span class="hex-text">{{ bgService.solidColor() }}</span>
                  </div>
                </div>
              }

              <!-- Gradient Controls -->
              @if (bgService.bgType() === 'gradient') {
                <div class="gradient-settings-box">
                  <div class="gradient-presets">
                    @for (g of presetGradients; track g.name) {
                      <button
                        type="button"
                        class="grad-chip"
                        [style.background]="'linear-gradient(135deg, ' + g.c1 + ', ' + g.c2 + ')'"
                        (click)="setGradient(g.c1, g.c2)"
                      ></button>
                    }
                  </div>
                  <div class="grad-color-row">
                    <div>
                      <label>Color 1:</label>
                      <input
                        type="color"
                        [value]="bgService.gradientColor1()"
                        (input)="onGrad1Change($event)"
                      />
                    </div>
                    <div>
                      <label>Color 2:</label>
                      <input
                        type="color"
                        [value]="bgService.gradientColor2()"
                        (input)="onGrad2Change($event)"
                      />
                    </div>
                  </div>
                </div>
              }
            </div>

            <!-- Export Buttons & Quality Selector -->
            <div class="export-card">
              <h3 class="card-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="7 10 12 15 17 10"></polyline>
                  <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                Download Cutout
              </h3>

              <!-- Custom Dark Popover Quality Selector Dropdown -->
              <div class="quality-selector-box" #dropdownContainer>
                <label class="quality-label">Download Quality</label>
                <div 
                  class="custom-dropdown-trigger" 
                  [class.open]="dropdownOpen"
                  (click)="toggleDropdown($event)"
                >
                  <div class="selected-option-label">
                    <span class="option-title">{{ getQualityLabel() }}</span>
                    @if (bgService.exportQuality() !== 'original') {
                      <span class="badge-ai">AI</span>
                    }
                  </div>
                  <svg class="chevron-icon" [class.rotated]="dropdownOpen" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="6 9 12 15 18 9"></polyline>
                  </svg>
                </div>

                @if (dropdownOpen) {
                  <div class="custom-dropdown-menu">
                    <div 
                      class="dropdown-option" 
                      [class.active]="bgService.exportQuality() === 'original'"
                      (click)="selectQuality('original')"
                    >
                      <div class="opt-text">
                        <span class="opt-title">Original Quality</span>
                        <span class="opt-desc">Native resolution, instant lossless export</span>
                      </div>
                      @if (bgService.exportQuality() === 'original') {
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2.5">
                          <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                      }
                    </div>

                    <div 
                      class="dropdown-option" 
                      [class.active]="bgService.exportQuality() === 'hd2x'"
                      (click)="selectQuality('hd2x')"
                    >
                      <div class="opt-text">
                        <span class="opt-title">HD 2× — AI Super Resolution</span>
                        <span class="opt-desc">2× detail recovery & sharp edges</span>
                      </div>
                      @if (bgService.exportQuality() === 'hd2x') {
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2.5">
                          <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                      }
                    </div>

                    <div 
                      class="dropdown-option" 
                      [class.active]="bgService.exportQuality() === 'hd4x'"
                      (click)="selectQuality('hd4x')"
                    >
                      <div class="opt-text">
                        <span class="opt-title">Ultra HD 4× — AI Super Resolution</span>
                        <span class="opt-desc">4× maximum AI detail recovery</span>
                      </div>
                      @if (bgService.exportQuality() === 'hd4x') {
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2.5">
                          <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                      }
                    </div>
                  </div>
                }
              </div>

              <!-- Dynamic Resolution Badge -->
              @if (bgService.targetExportDimensions(); as dims) {
                <div class="export-dim-badge">
                  <span>Output resolution: <strong>{{ dims.width }} × {{ dims.height }} px</strong> ({{ dims.label }})</span>
                </div>
              }



              <!-- Helper note for upscaling -->
              @if (bgService.exportQuality() !== 'original') {
                <p class="export-hint-note">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                  </svg>
                  <span>High-quality upscaling may take a moment.</span>
                </p>
              }

              <div class="export-grid">
                <button
                  class="btn-download-primary"
                  [disabled]="bgService.isExporting() || bgService.isProcessing()"
                  (click)="bgService.downloadResult('png')"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                  </svg>
                  {{ bgService.isExporting() ? 'Exporting...' : 'Download PNG (Alpha)' }}
                </button>
                <div class="export-sub-row">
                  <button
                    class="btn-download-sec"
                    [disabled]="bgService.isExporting() || bgService.isProcessing()"
                    (click)="bgService.downloadResult('webp')"
                  >
                    Download WEBP
                  </button>
                  <button
                    class="btn-download-sec"
                    [disabled]="bgService.isExporting() || bgService.isProcessing()"
                    (click)="bgService.downloadResult('jpg')"
                  >
                    Download JPG
                  </button>
                </div>
              </div>

              <p class="export-notice">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="16" x2="12" y2="12"></line>
                  <line x1="12" y1="8" x2="12.01" y2="8"></line>
                </svg>
                <span>JPG does not support transparency. Solid background will be applied automatically.</span>
              </p>
            </div>
          </div>
        </div>
      }
    </div>
  `,
  styles: [`
    .bgr-shell {
      width: 100%;
      max-width: 1320px;
      margin: 0 auto;
      padding: 24px 20px 48px;
    }

    .privacy-banner-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 14px 20px;
      margin-bottom: 24px;
      display: flex;
      align-items: center;
      gap: 16px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }

    .privacy-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      background: var(--success-bg);
      color: var(--success-text);
      border: 1px solid var(--success-border);
      padding: 6px 12px;
      border-radius: var(--radius-pill);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }

    .privacy-desc {
      font-size: 13px;
      color: var(--text-muted);
      margin: 0;
      line-height: 1.5;
    }

    /* Dropzone */
    .bgr-dropzone {
      background: var(--bg-card);
      border: 2px dashed var(--border-hover);
      border-radius: var(--radius-lg);
      padding: 64px 24px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 14px;
    }

    .bgr-dropzone:hover, .bgr-dropzone.dragover {
      border-color: var(--primary);
      background: var(--primary-light);
      transform: translateY(-2px);
    }

    .drop-icon-box {
      width: 80px;
      height: 80px;
      border-radius: 50%;
      background: var(--primary-light);
      color: var(--primary);
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 8px;
    }

    .drop-title {
      font-family: 'Outfit', sans-serif;
      font-size: 24px;
      font-weight: 800;
      color: var(--text-main);
      margin: 0;
    }

    .drop-subtitle {
      font-size: 14px;
      color: var(--text-muted);
      margin: 0;
    }

    .btn-select-image {
      margin-top: 10px;
      background: var(--primary-gradient);
      color: #fff;
      border: none;
      padding: 12px 28px;
      border-radius: var(--radius-sm);
      font-weight: 700;
      font-size: 15px;
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
      transition: transform 0.15s ease;
    }

    .btn-select-image:hover {
      transform: scale(1.03);
    }

    /* Workbench Layout */
    .bgr-workbench-layout {
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 24px;
      align-items: start;
    }

    @media (max-width: 980px) {
      .bgr-workbench-layout {
        grid-template-columns: 1fr;
      }
    }

    .workbench-canvas-panel {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .canvas-header-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }

    .file-info-tag {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .file-name {
      font-weight: 700;
      font-size: 14px;
      color: var(--text-main);
    }

    .dims-badge {
      font-size: 11px;
      background: var(--bg-main);
      border: 1px solid var(--border-subtle);
      padding: 2px 8px;
      border-radius: var(--radius-pill);
      color: var(--text-muted);
    }

    .perf-pill {
      font-size: 12px;
      font-weight: 600;
      color: var(--success-text);
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .dot-live {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
    }

    /* Interactive Comparison Container */
    .comparison-container {
      position: relative;
      width: 100%;
      height: 520px;
      border-radius: var(--radius-md);
      overflow: hidden;
      user-select: none;
      -webkit-user-select: none;
      border: 1px solid var(--border-subtle);
      cursor: ew-resize;
      background: #0f172a;
    }

    .canvas-bg-underlay {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1;
    }

    .canvas-bg-underlay.transparent {
      background-image: linear-gradient(45deg, #e5e7eb 25%, transparent 25%),
        linear-gradient(-45deg, #e5e7eb 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #e5e7eb 75%),
        linear-gradient(-45deg, transparent 75%, #e5e7eb 75%);
      background-size: 20px 20px;
      background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
      background-color: #ffffff;
    }

    .original-clip-layer {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 2;
      background: transparent;
      will-change: clip-path;
      pointer-events: none;
    }

    .comparison-img {
      max-width: 100%;
      max-height: 100%;
      width: auto;
      height: auto;
      object-fit: contain;
      pointer-events: none;
      display: block;
    }

    .split-divider-handle {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 2px;
      background: #ffffff;
      z-index: 10;
      transform: translateX(-50%);
      pointer-events: none;
      box-shadow: 0 0 8px rgba(0, 0, 0, 0.4);
    }

    .handle-pill {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 36px;
      height: 36px;
      background: #ffffff;
      color: #0f172a;
      border-radius: 50%;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: ew-resize;
      pointer-events: auto;
    }

    .split-tag {
      position: absolute;
      bottom: 12px;
      padding: 5px 12px;
      border-radius: var(--radius-pill);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      backdrop-filter: blur(8px);
      z-index: 15;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }

    .original-tag {
      left: 12px;
      background: rgba(0, 0, 0, 0.65);
      color: #ffffff;
    }

    .result-tag {
      right: 12px;
      background: rgba(37, 99, 235, 0.85);
      color: #ffffff;
    }

    .processing-overlay {
      position: absolute;
      inset: 0;
      background: rgba(15, 23, 42, 0.75);
      backdrop-filter: blur(6px);
      z-index: 30;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }

    .scanner-laser {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: #38bdf8;
      box-shadow: 0 0 16px #38bdf8;
      animation: scanLaser 2s ease-in-out infinite alternate;
    }

    @keyframes scanLaser {
      0% { top: 5%; }
      100% { top: 95%; }
    }

    .spinner-box {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 14px;
      color: #ffffff;
    }

    .cv-spinner {
      width: 42px;
      height: 42px;
      border: 3px solid rgba(255, 255, 255, 0.2);
      border-top-color: #38bdf8;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .processing-label {
      font-size: 14px;
      font-weight: 600;
    }

    .error-banner {
      position: absolute;
      inset: 0;
      background: rgba(239, 68, 68, 0.95);
      color: #ffffff;
      padding: 24px;
      z-index: 30;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      text-align: center;
    }

    .btn-retry {
      background: #ffffff;
      color: #ef4444;
      border: none;
      padding: 8px 18px;
      border-radius: var(--radius-sm);
      font-weight: 700;
      cursor: pointer;
    }

    .slider-control-row {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 13px;
      color: var(--text-muted);
    }

    .custom-range {
      flex: 1;
      accent-color: var(--primary);
    }

    .slider-val {
      font-weight: 700;
      min-width: 40px;
    }

    /* Right Sidebar Controls */
    .workbench-controls-panel {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .action-top-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .btn-tool-secondary {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 10px;
      border-radius: var(--radius-sm);
      font-weight: 600;
      font-size: 13px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      cursor: pointer;
    }

    .btn-reprocess {
      background: var(--primary-light);
      color: var(--primary);
      border: 1px solid var(--border-subtle);
      padding: 10px;
      border-radius: var(--radius-sm);
      font-weight: 700;
      font-size: 13px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      cursor: pointer;
    }

    .control-card, .export-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .card-title {
      font-family: 'Outfit', sans-serif;
      font-size: 15px;
      font-weight: 700;
      color: var(--text-main);
      margin: 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .slider-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .slider-header {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 600;
      color: var(--text-main);
    }

    .control-hint {
      font-size: 11px;
      color: var(--text-muted);
    }

    .tuning-actions-row {
      display: grid;
      grid-template-columns: 1fr 1.6fr;
      gap: 10px;
      margin-top: 8px;
      padding-top: 14px;
      border-top: 1px solid var(--border-subtle);
    }

    .btn-tuning-reset {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      padding: 10px 14px;
      background: var(--bg-card);
      border: 1px solid var(--border-hover);
      color: var(--text-body);
      border-radius: var(--radius-sm);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .btn-tuning-reset:hover:not(:disabled) {
      background: var(--bg-subtle);
      border-color: var(--text-muted);
      color: var(--text-main);
      transform: translateY(-1px);
    }

    .btn-tuning-reset:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .btn-tuning-proceed {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 10px 16px;
      background: var(--primary-gradient);
      color: #ffffff;
      border: none;
      border-radius: var(--radius-sm);
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 4px 14px var(--primary-glow);
      transition: all 0.2s ease;
    }

    .btn-tuning-proceed:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 6px 18px var(--primary-glow);
      filter: brightness(1.08);
    }

    .btn-tuning-proceed:active:not(:disabled) {
      transform: translateY(0);
    }

    .btn-tuning-proceed:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }

    .btn-spinner {
      width: 14px;
      height: 14px;
      border: 2px solid rgba(255, 255, 255, 0.35);
      border-top-color: #ffffff;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      display: inline-block;
    }

    .unprocessed-guide-badge {
      position: absolute;
      top: 14px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(8px);
      color: #ffffff;
      padding: 8px 16px;
      border-radius: var(--radius-pill);
      font-size: 12px;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
      z-index: 20;
      pointer-events: none;
      white-space: nowrap;
    }

    .bg-mode-tabs {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 4px;
      background: var(--bg-main);
      padding: 4px;
      border-radius: var(--radius-sm);
    }

    .mode-tab {
      background: transparent;
      border: none;
      padding: 7px 4px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      cursor: pointer;
      text-align: center;
    }

    .mode-tab.active {
      background: var(--bg-card);
      color: var(--primary);
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    }

    .color-presets, .gradient-presets {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 8px;
    }

    .color-chip, .grad-chip {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      border: 2px solid transparent;
      cursor: pointer;
    }

    .color-chip.selected {
      border-color: var(--primary);
      transform: scale(1.1);
    }

    .custom-color-picker {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 10px;
      font-size: 12px;
      font-weight: 600;
    }

    .color-input {
      width: 32px;
      height: 32px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }

    .grad-color-row {
      display: flex;
      gap: 16px;
      margin-top: 10px;
      font-size: 12px;
    }

    .quality-selector-box {
      position: relative;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .quality-label {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-main);
    }

    .custom-dropdown-trigger {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      padding: 10px 14px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-subtle);
      background: #0f172a;
      color: #f8fafc;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      user-select: none;
      transition: all 0.2s ease;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    .custom-dropdown-trigger:hover, .custom-dropdown-trigger.open {
      border-color: #3b82f6;
      background: #1e293b;
    }

    .selected-option-label {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .badge-ai {
      background: #3b82f6;
      color: #ffffff;
      font-size: 10px;
      font-weight: 800;
      padding: 2px 6px;
      border-radius: 4px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }

    .chevron-icon {
      transition: transform 0.2s ease;
      color: #94a3b8;
    }

    .chevron-icon.rotated {
      transform: rotate(180deg);
      color: #3b82f6;
    }

    .custom-dropdown-menu {
      position: absolute;
      top: calc(100% + 6px);
      left: 0;
      right: 0;
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: var(--radius-sm);
      box-shadow: 0 12px 28px rgba(0, 0, 0, 0.45);
      z-index: 100;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      animation: dropdownFadeIn 0.15s ease-out;
    }

    @keyframes dropdownFadeIn {
      from { opacity: 0; transform: translateY(-4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .dropdown-option {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 14px;
      cursor: pointer;
      transition: background 0.15s ease;
      border-bottom: 1px solid #1e293b;
    }

    .dropdown-option:last-child {
      border-bottom: none;
    }

    .dropdown-option:hover {
      background: #1e293b;
    }

    .dropdown-option.active {
      background: #1e293b;
    }

    .opt-text {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .opt-title {
      font-size: 13px;
      font-weight: 700;
      color: #f8fafc;
    }

    .dropdown-option.active .opt-title {
      color: #60a5fa;
    }

    .opt-desc {
      font-size: 11px;
      color: #94a3b8;
    }

    .export-dim-badge {
      font-size: 12px;
      color: var(--text-muted);
      background: var(--bg-main);
      padding: 6px 12px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-subtle);
    }

    .export-dim-badge strong {
      color: var(--text-main);
    }

    .export-hint-note {
      font-size: 11px;
      color: #818cf8;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 6px;
      line-height: 1.3;
    }

    .export-hint-note svg {
      flex-shrink: 0;
    }

    .export-notice {
      font-size: 11px;
      color: var(--text-muted);
      margin: 2px 0 0;
      display: flex;
      align-items: flex-start;
      gap: 6px;
      line-height: 1.4;
    }

    .export-notice svg {
      flex-shrink: 0;
      margin-top: 2px;
      color: var(--primary);
    }

    .export-grid {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .btn-download-primary {
      background: var(--primary-gradient);
      color: #ffffff;
      border: none;
      padding: 13px;
      border-radius: var(--radius-sm);
      font-weight: 700;
      font-size: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    .export-sub-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .btn-download-sec {
      background: var(--bg-main);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 9px;
      border-radius: var(--radius-sm);
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }

    .perf-metrics-card {
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: var(--radius-md);
      padding: 12px 14px;
      margin: 12px 0 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
    }

    .metric-label {
      color: #94a3b8;
    }

    .metric-val {
      color: #f8fafc;
      font-weight: 600;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    .metric-val.highlight {
      color: #34d399;
    }

    .metric-val.highlight-export {
      color: #60a5fa;
    }
  `]
})
export class BgRemoverComponent {
  bgService = inject(BgRemoverService);

  @ViewChild('cmpContainer') cmpContainer?: ElementRef<HTMLDivElement>;
  @ViewChild('dropdownContainer') dropdownContainer?: ElementRef<HTMLDivElement>;

  isDragging = false;
  dropdownOpen = false;

  toggleDropdown(e?: Event): void {
    if (e) e.stopPropagation();
    this.dropdownOpen = !this.dropdownOpen;
  }

  selectQuality(quality: 'original' | 'hd2x' | 'hd4x'): void {
    this.bgService.exportQuality.set(quality);
    this.dropdownOpen = false;
  }

  getQualityLabel(): string {
    const q = this.bgService.exportQuality();
    if (q === 'hd4x') return 'Ultra HD 4× — AI Super Resolution';
    if (q === 'hd2x') return 'HD 2× — AI Super Resolution';
    return 'Original Quality';
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    if (this.dropdownOpen && this.dropdownContainer && !this.dropdownContainer.nativeElement.contains(event.target as Node)) {
      this.dropdownOpen = false;
    }
  }

  @HostListener('document:keydown.escape')
  onEscapeKey(): void {
    this.dropdownOpen = false;
  }

  presetColors = ['#ffffff', '#000000', '#f3f4f6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  presetGradients = [
    { name: 'Indigo-Pink', c1: '#6366f1', c2: '#ec4899' },
    { name: 'Cyan-Blue', c1: '#06b6d4', c2: '#3b82f6' },
    { name: 'Emerald-Teal', c1: '#10b981', c2: '#14b8a6' },
    { name: 'Amber-Rose', c1: '#f59e0b', c2: '#f43f5e' },
    { name: 'Dark-Slate', c1: '#1e293b', c2: '#0f172a' },
  ];

  onDragOver(e: DragEvent): void {
    e.preventDefault();
    e.stopPropagation();
    this.isDragging = true;
  }

  onDragLeave(e: DragEvent): void {
    e.preventDefault();
    e.stopPropagation();
    this.isDragging = false;
  }

  onDrop(e: DragEvent): void {
    e.preventDefault();
    e.stopPropagation();
    this.isDragging = false;
    if (e.dataTransfer?.files?.length) {
      this.bgService.setImage(e.dataTransfer.files[0]);
    }
  }

  onFileSelected(e: Event): void {
    const input = e.target as HTMLInputElement;
    if (input.files?.length) {
      this.bgService.setImage(input.files[0]);
    }
  }

  onBgFileSelected(e: Event): void {
    const input = e.target as HTMLInputElement;
    if (input.files?.length) {
      this.bgService.setCustomBgImage(input.files[0]);
    }
  }

  onQualityChange(e: Event): void {
    const val = (e.target as HTMLSelectElement).value as 'original' | 'hd2x' | 'hd4x';
    this.bgService.exportQuality.set(val);
  }

  onSplitChange(e: Event): void {
    const val = Number((e.target as HTMLInputElement).value);
    this.bgService.splitPosition.set(val);
  }

  isSliderDragging = false;

  onMouseDown(e: MouseEvent): void {
    this.isSliderDragging = true;
    this.updateSplitFromPointer(e.clientX);
  }

  onMouseMove(e: MouseEvent): void {
    if ((this.isSliderDragging || e.buttons === 1) && this.cmpContainer) {
      this.updateSplitFromPointer(e.clientX);
    }
  }

  onMouseUp(): void {
    this.isSliderDragging = false;
  }

  onMouseLeave(e: MouseEvent): void {
    if (e.buttons !== 1) {
      this.isSliderDragging = false;
    }
  }

  @HostListener('window:mouseup')
  onWindowMouseUp(): void {
    this.isSliderDragging = false;
  }

  onTouchStart(e: TouchEvent): void {
    this.isSliderDragging = true;
    if (e.touches.length) {
      this.updateSplitFromPointer(e.touches[0].clientX);
    }
  }

  onTouchMove(e: TouchEvent): void {
    if (!this.cmpContainer || !e.touches.length) return;
    this.updateSplitFromPointer(e.touches[0].clientX);
  }

  onTouchEnd(): void {
    this.isSliderDragging = false;
  }

  private updateSplitFromPointer(clientX: number): void {
    if (!this.cmpContainer) return;
    const rect = this.cmpContainer.nativeElement.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    this.bgService.splitPosition.set(Math.round(pct));
  }

  onParamInput(param: 'sensitivity' | 'edgeSoftness' | 'defringeStrength', e: Event): void {
    const val = Math.round(Number((e.target as HTMLInputElement).value));
    if (param === 'sensitivity') this.bgService.sensitivity.set(val);
    if (param === 'edgeSoftness') this.bgService.edgeSoftness.set(val);
    if (param === 'defringeStrength') this.bgService.defringeStrength.set(val);
  }

  onColorChange(e: Event): void {
    this.bgService.solidColor.set((e.target as HTMLInputElement).value);
  }

  onGrad1Change(e: Event): void {
    this.bgService.gradientColor1.set((e.target as HTMLInputElement).value);
  }

  onGrad2Change(e: Event): void {
    this.bgService.gradientColor2.set((e.target as HTMLInputElement).value);
  }

  setGradient(c1: string, c2: string): void {
    this.bgService.gradientColor1.set(c1);
    this.bgService.gradientColor2.set(c2);
    this.bgService.bgType.set('gradient');
  }

  getUnderlayBackground(): string | null {
    const type = this.bgService.bgType();
    if (type === 'gradient') {
      return `linear-gradient(135deg, ${this.bgService.gradientColor1()}, ${this.bgService.gradientColor2()})`;
    }
    if (type === 'image' && this.bgService.customBgImageUrl()) {
      return `url('${this.bgService.customBgImageUrl()}') center / cover no-repeat`;
    }
    return null;
  }
}
