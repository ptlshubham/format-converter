import { CommonModule } from '@angular/common';
import { Component, ElementRef, HostListener, ViewChild, inject } from '@angular/core';
import { ConversionStateService } from '../../services/conversion-state.service';
import { SupportedFormat } from '../../models/image-item.model';

@Component({
  selector: 'app-landing',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div
      class="landing-3d-scene"
      [class.drag-over]="isDraggingOver"
      (dragover)="onDragOver($event)"
      (dragleave)="onDragLeave($event)"
      (drop)="onDrop($event)"
    >
      <!-- 3D Ambient Floating Geometric Shapes in Background -->
      <div class="ambient-3d-shapes" aria-hidden="true">
        <div class="shape-3d cube-1">
          <div class="cube-face front"></div>
          <div class="cube-face back"></div>
          <div class="cube-face right"></div>
          <div class="cube-face left"></div>
          <div class="cube-face top"></div>
          <div class="cube-face bottom"></div>
        </div>
        <div class="shape-3d cube-2">
          <div class="cube-face front"></div>
          <div class="cube-face back"></div>
          <div class="cube-face right"></div>
          <div class="cube-face left"></div>
          <div class="cube-face top"></div>
          <div class="cube-face bottom"></div>
        </div>
        <div class="shape-3d orb-1"></div>
        <div class="shape-3d orb-2"></div>
      </div>

      <div class="landing-stage">
        <!-- 3D Target Format Pill Bar -->
        <div class="format-pill-bar-3d">
          <span class="pill-label-3d">Convert to:</span>
          <div class="pills-track-3d">
            @for (fmt of state.supportedFormats; track fmt.id) {
              <button
                type="button"
                class="format-pill-3d"
                [class.active]="state.targetFormat() === fmt.id"
                (click)="state.setTargetFormat(fmt.id)"
              >
                {{ fmt.id.toUpperCase() }}
              </button>
            }
          </div>
        </div>

        <!-- 3D Hero Headline -->
        <div class="hero-3d-header">
          <h1 class="hero-3d-title">
            Convert Files to
            <span class="format-3d-glow">{{ currentFormatLabel() }}</span>
          </h1>
          <p class="hero-3d-subtitle">
            Next-gen 3D client-side conversion engine. Convert Images, PDF, SVG, and RAW instantly with zero server uploads.
          </p>
        </div>

        <!-- Interactive 3D Parallax Dropzone Card -->
        <div class="dropzone-3d-wrapper">
          <div
            #tiltCard
            class="dropzone-3d-card"
            [style.transform]="cardTransform"
            (mousemove)="onCardMouseMove($event)"
            (mouseleave)="onCardMouseLeave()"
            (click)="triggerFileInput()"
          >
            <!-- Dynamic Specular Glare Reflection -->
            <div
              class="card-glare"
              [style.background]="glareBackground"
            ></div>

            <div class="card-3d-inner">
              <!-- Floating 3D Upload Icon -->
              <div class="upload-icon-3d">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
              </div>

              <!-- 3D Push Button -->
              <button
                type="button"
                class="btn-3d-primary btn-select-3d"
                (click)="$event.stopPropagation(); triggerFileInput()"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <circle cx="8.5" cy="8.5" r="1.5"></circle>
                  <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
                <span>Select Files (Images & PDF)</span>
              </button>

              <p class="dropzone-3d-hint">or drop photos and PDFs anywhere on this 3D stage</p>
            </div>

            <!-- Supported Format Pill Tags in 3D -->
            <div class="format-tags-3d">
              <span class="tag-item">PDF</span>
              <span class="dot">•</span>
              <span class="tag-item">JPG</span>
              <span class="dot">•</span>
              <span class="tag-item">PNG</span>
              <span class="dot">•</span>
              <span class="tag-item">WEBP</span>
              <span class="dot">•</span>
              <span class="tag-item">GIF</span>
              <span class="dot">•</span>
              <span class="tag-item">SVG</span>
              <span class="dot">•</span>
              <span class="tag-item">HEIC / RAW</span>
            </div>
          </div>
        </div>

        <!-- Hidden multi-file input -->
        <input
          #fileInput
          type="file"
          multiple
          accept="image/*,.png,.jpg,.jpeg,.webp,.gif,.tif,.tiff,.psd,.svg,.bmp,.ico,.heic,.heif,.raw,.cr2,.nef,.arw,.dng,.orf,.pdf,.txt"
          style="display: none"
          (change)="onFilesSelected($event)"
        />

        <!-- 3D Feature Pillars -->
        <div class="features-3d-grid">
          <div class="feature-card-3d">
            <div class="feat-pod-3d">⚡</div>
            <div class="feat-details-3d">
              <h3 class="feat-heading-3d">Lightning Fast</h3>
              <p class="feat-text-3d">Local GPU/CPU processing with zero upload queue latency.</p>
            </div>
          </div>

          <div class="feature-card-3d">
            <div class="feat-pod-3d">🔒</div>
            <div class="feat-details-3d">
              <h3 class="feat-heading-3d">100% Private</h3>
              <p class="feat-text-3d">Your photos never touch any server. Total local security.</p>
            </div>
          </div>

          <div class="feature-card-3d">
            <div class="feat-pod-3d">📦</div>
            <div class="feat-details-3d">
              <h3 class="feat-heading-3d">Batch ZIP Export</h3>
              <p class="feat-text-3d">Convert hundreds of photos at once and download as one ZIP.</p>
            </div>
          </div>
        </div>
      </div>

      <!-- 3D Drag Over Modal -->
      @if (isDraggingOver) {
        <div class="drag-active-3d-overlay">
          <div class="drag-active-3d-card">
            <div class="drag-3d-icon">
              <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </div>
            <h2>Drop Images into 3D Space</h2>
            <p>Target format: <strong>{{ currentFormatLabel() }}</strong></p>
          </div>
        </div>
      }
    </div>
  `,
  styles: [`
    .landing-3d-scene {
      min-height: calc(100vh - 120px);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 40px 20px 60px;
      position: relative;
      perspective: 1200px;
      overflow: hidden;
    }

    /* 3D Floating Ambient Shapes */
    .ambient-3d-shapes {
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 0;
      overflow: hidden;
    }

    .shape-3d {
      position: absolute;
      transform-style: preserve-3d;
    }

    /* CSS 3D Cube */
    .cube-1 {
      top: 15%;
      left: 8%;
      width: 50px;
      height: 50px;
      animation: float3D 8s infinite ease-in-out, rotateCube 16s infinite linear;
      opacity: 0.6;
    }

    .cube-2 {
      bottom: 20%;
      right: 8%;
      width: 65px;
      height: 65px;
      animation: float3DAlt 10s infinite ease-in-out, rotateCube 20s infinite linear reverse;
      opacity: 0.5;
    }

    .cube-face {
      position: absolute;
      width: 100%;
      height: 100%;
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(37, 99, 235, 0.1));
      border: 1px solid rgba(59, 130, 246, 0.4);
      backdrop-filter: blur(4px);
    }

    .cube-1 .cube-face.front  { transform: rotateY(0deg) translateZ(25px); }
    .cube-1 .cube-face.back   { transform: rotateY(180deg) translateZ(25px); }
    .cube-1 .cube-face.right  { transform: rotateY(90deg) translateZ(25px); }
    .cube-1 .cube-face.left   { transform: rotateY(-90deg) translateZ(25px); }
    .cube-1 .cube-face.top    { transform: rotateX(90deg) translateZ(25px); }
    .cube-1 .cube-face.bottom { transform: rotateX(-90deg) translateZ(25px); }

    .cube-2 .cube-face.front  { transform: rotateY(0deg) translateZ(32px); }
    .cube-2 .cube-face.back   { transform: rotateY(180deg) translateZ(32px); }
    .cube-2 .cube-face.right  { transform: rotateY(90deg) translateZ(32px); }
    .cube-2 .cube-face.left   { transform: rotateY(-90deg) translateZ(32px); }
    .cube-2 .cube-face.top    { transform: rotateX(90deg) translateZ(32px); }
    .cube-2 .cube-face.bottom { transform: rotateX(-90deg) translateZ(32px); }

    @keyframes rotateCube {
      from { transform: rotateX(0deg) rotateY(0deg) rotateZ(0deg); }
      to { transform: rotateX(360deg) rotateY(360deg) rotateZ(360deg); }
    }

    .orb-1 {
      top: 10%;
      right: 15%;
      width: 180px;
      height: 180px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0) 70%);
      filter: blur(20px);
      animation: float3D 9s infinite ease-in-out;
    }

    .orb-2 {
      bottom: 12%;
      left: 14%;
      width: 220px;
      height: 220px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(96, 165, 250, 0.12) 0%, rgba(37, 99, 235, 0) 70%);
      filter: blur(25px);
      animation: float3DAlt 11s infinite ease-in-out;
    }

    .landing-stage {
      width: 100%;
      max-width: 880px;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      z-index: 1;
      position: relative;
    }

    /* 3D Format Pill Bar */
    .format-pill-bar-3d {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      background: var(--bg-card);
      padding: 8px 14px;
      border-radius: var(--radius-pill);
      border: 1px solid var(--border-subtle);
      box-shadow: 0 4px 0 var(--border-hover), 0 8px 18px rgba(0, 0, 0, 0.06);
      margin-bottom: 28px;
      flex-wrap: wrap;
      justify-content: center;
      transition: all 0.3s ease;
      max-width: 100%;
    }

    .pill-label-3d {
      font-size: 12px;
      font-weight: 800;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.6px;
      padding-left: 4px;
    }

    .pills-track-3d {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: center;
    }

    .format-pill-3d {
      font-size: 12px;
      font-weight: 800;
      padding: 6px 14px;
      border-radius: var(--radius-pill);
      background: var(--bg-subtle);
      color: var(--text-body);
      border: 1px solid var(--border-subtle);
      box-shadow: 0 2px 0 var(--border-hover);
      transform: translateY(0);
      transition: all 0.12s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .format-pill-3d:hover {
      transform: translateY(-1px);
      box-shadow: 0 3px 0 var(--border-hover);
      color: var(--text-main);
    }

    .format-pill-3d:active {
      transform: translateY(2px);
      box-shadow: 0 0 0 var(--border-hover);
    }

    .format-pill-3d.active {
      background: var(--primary-gradient);
      color: #ffffff;
      border-color: var(--primary-hover);
      box-shadow: 0 3px 0 var(--3d-btn-primary-edge), 0 6px 14px rgba(37, 99, 235, 0.35);
    }

    /* 3D Hero Title */
    .hero-3d-header {
      margin-bottom: 34px;
      width: 100%;
      padding: 0 10px;
    }

    .hero-3d-title {
      font-size: clamp(28px, 5.5vw, 44px);
      font-weight: 900;
      color: var(--text-main);
      line-height: 1.18;
      margin-bottom: 12px;
      letter-spacing: -0.6px;
    }

    .format-3d-glow {
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      filter: drop-shadow(0 4px 16px rgba(59, 130, 246, 0.4));
      display: inline-block;
      position: relative;
    }

    .hero-3d-subtitle {
      font-size: clamp(14px, 2vw, 16px);
      color: var(--text-muted);
      line-height: 1.6;
      max-width: 620px;
      margin: 0 auto;
    }

    /* 3D Parallax Dropzone Card */
    .dropzone-3d-wrapper {
      width: 100%;
      perspective: 1000px;
      margin-bottom: 36px;
    }

    .dropzone-3d-card {
      width: 100%;
      background: var(--bg-card);
      border: 2px dashed var(--border-hover);
      border-radius: var(--radius-xl);
      padding: clamp(32px, 5vw, 50px) clamp(16px, 4vw, 36px);
      cursor: pointer;
      box-shadow: var(--3d-card-shadow);
      transform-style: preserve-3d;
      transition: box-shadow 0.25s ease, border-color 0.25s ease, transform 0.15s ease-out;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 20px;
      position: relative;
      overflow: hidden;
    }

    .dropzone-3d-card:hover {
      border-color: var(--primary);
      box-shadow: var(--3d-card-shadow-hover);
    }

    .card-glare {
      position: absolute;
      inset: 0;
      pointer-events: none;
      border-radius: var(--radius-xl);
      transition: background 0.1s ease;
      z-index: 10;
    }

    .card-3d-inner {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
      transform-style: preserve-3d;
      z-index: 5;
    }

    .upload-icon-3d {
      width: clamp(56px, 10vw, 72px);
      height: clamp(56px, 10vw, 72px);
      border-radius: 50%;
      background: var(--primary-light);
      color: var(--primary);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 6px 0 var(--border-subtle), 0 10px 20px rgba(59, 130, 246, 0.2);
      transform: translateZ(40px);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      position: relative;
    }

    .dropzone-3d-card:hover .upload-icon-3d {
      transform: translateZ(55px) scale(1.08);
      box-shadow: 0 8px 0 var(--3d-btn-primary-edge), 0 14px 28px rgba(37, 99, 235, 0.35);
    }

    .btn-select-3d {
      font-size: clamp(16px, 2.5vw, 19px);
      padding: clamp(12px, 2vw, 16px) clamp(24px, 5vw, 44px);
      gap: 10px;
      transform: translateZ(50px);
      letter-spacing: 0.3px;
    }

    .dropzone-3d-card:hover .btn-select-3d {
      transform: translateZ(65px);
    }

    .dropzone-3d-hint {
      font-size: 14px;
      color: var(--text-muted);
      font-weight: 500;
      transform: translateZ(25px);
    }

    .format-tags-3d {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--text-light);
      font-weight: 600;
      flex-wrap: wrap;
      justify-content: center;
      transform: translateZ(20px);
    }

    .tag-item {
      background: var(--bg-subtle);
      padding: 3px 8px;
      border-radius: var(--radius-xs);
      border: 1px solid var(--border-subtle);
      box-shadow: 0 1px 0 var(--border-hover);
    }

    /* 3D Features Grid */
    .features-3d-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
      width: 100%;
    }

    .feature-card-3d {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 20px;
      text-align: left;
      display: flex;
      align-items: flex-start;
      gap: 14px;
      box-shadow: 0 5px 0 var(--border-subtle), 0 10px 20px rgba(0, 0, 0, 0.04);
      transform: translateY(0);
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .feature-card-3d:hover {
      transform: translateY(-4px);
      box-shadow: 0 8px 0 var(--border-hover), 0 16px 30px rgba(0, 0, 0, 0.08);
      border-color: var(--border-hover);
    }

    .feat-pod-3d {
      font-size: 22px;
      line-height: 1;
      padding: 8px;
      background: var(--bg-subtle);
      border-radius: var(--radius-md);
      border: 1px solid var(--border-subtle);
      box-shadow: 0 3px 0 var(--border-hover);
    }

    .feat-details-3d {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }

    .feat-heading-3d {
      font-size: 15px;
      font-weight: 800;
      color: var(--text-main);
    }

    .feat-text-3d {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.45;
    }

    /* 3D Drag Active Overlay */
    .drag-active-3d-overlay {
      position: fixed;
      inset: 0;
      background: var(--bg-modal);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: center;
      pointer-events: none;
      perspective: 1000px;
      padding: 20px;
    }

    .drag-active-3d-card {
      background: var(--bg-card);
      border: 3px dashed var(--primary);
      border-radius: var(--radius-xl);
      padding: clamp(32px, 6vw, 50px) clamp(24px, 6vw, 70px);
      text-align: center;
      box-shadow: 0 25px 60px rgba(37, 99, 235, 0.4), 0 8px 0 var(--3d-btn-primary-edge);
      transform: rotateX(8deg) scale(1.04);
      animation: pulseGlow 1.8s infinite ease-in-out;
      max-width: 90vw;
    }

    .drag-3d-icon {
      color: var(--primary);
      margin-bottom: 14px;
      filter: drop-shadow(0 4px 12px rgba(59, 130, 246, 0.5));
    }

    .drag-active-3d-card h2 {
      font-size: clamp(20px, 4vw, 28px);
      font-weight: 900;
      color: var(--text-main);
    }

    .drag-active-3d-card p {
      font-size: 14px;
      color: var(--text-muted);
      margin-top: 8px;
    }

    /* Responsive Breakpoints */
    @media (max-width: 900px) {
      .features-3d-grid {
        grid-template-columns: repeat(2, 1fr);
      }
    }

    @media (max-width: 640px) {
      .landing-3d-scene {
        padding: 24px 14px 40px;
      }
      .format-pill-bar-3d {
        padding: 6px 10px;
        gap: 6px;
      }
      .format-pill-3d {
        padding: 5px 10px;
        font-size: 11px;
      }
      .features-3d-grid {
        grid-template-columns: 1fr;
        gap: 12px;
      }
      .cube-1, .cube-2 {
        display: none;
      }
    }
  `]
})
export class LandingComponent {
  state = inject(ConversionStateService);
  isDraggingOver = false;

  cardTransform = 'perspective(1000px) rotateX(0deg) rotateY(0deg)';
  glareBackground = 'none';

  @ViewChild('fileInput') fileInputRef!: ElementRef<HTMLInputElement>;
  @ViewChild('tiltCard') tiltCardRef!: ElementRef<HTMLDivElement>;

  currentFormatLabel(): string {
    return this.state.targetFormat().toUpperCase();
  }

  setTargetFormat(fmt: SupportedFormat): void {
    this.state.setTargetFormat(fmt);
  }

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

  onCardMouseMove(event: MouseEvent): void {
    if (!this.tiltCardRef) return;
    const rect = this.tiltCardRef.nativeElement.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const rotX = ((y - centerY) / centerY) * -10;
    const rotY = ((x - centerX) / centerX) * 10;

    this.cardTransform = `perspective(1000px) rotateX(${rotX.toFixed(2)}deg) rotateY(${rotY.toFixed(2)}deg) scale(1.02)`;

    const glareX = (x / rect.width) * 100;
    const glareY = (y / rect.height) * 100;
    this.glareBackground = `radial-gradient(circle at ${glareX}% ${glareY}%, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0) 60%)`;
  }

  onCardMouseLeave(): void {
    this.cardTransform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale(1)';
    this.glareBackground = 'none';
  }

  @HostListener('window:dragover', ['$event'])
  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDraggingOver = true;
  }

  @HostListener('window:dragleave', ['$event'])
  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    if (event.clientX === 0 && event.clientY === 0) {
      this.isDraggingOver = false;
    }
  }

  @HostListener('window:drop', ['$event'])
  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDraggingOver = false;

    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
      this.state.addFiles(event.dataTransfer.files);
    }
  }
}
