import { CommonModule } from '@angular/common';
import { Component, inject } from '@angular/core';
import { ConversionStateService } from '../../services/conversion-state.service';
import { BgRemoverService } from '../../services/bg-remover.service';
import { ImageItem } from '../../models/image-item.model';

@Component({
  selector: 'app-result-screen',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="result-3d-wrapper">
      <div class="result-3d-container">
        <!-- 3D Success Header with Floating Trophy Badge -->
        <div class="success-3d-header">
          <div class="trophy-badge-3d">
            <div class="trophy-glow-ring"></div>
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"></path>
              <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"></path>
              <path d="M4 22h16"></path>
              <path d="M10 14.66V17c0 .55-.45 1-1 1H7v2h10v-2h-2c-.55 0-1-.45-1-1v-2.34"></path>
              <path d="M6 4h12v7a6 6 0 0 1-12 0V4z"></path>
            </svg>
          </div>
          <h1 class="success-3d-title">Conversion Complete!</h1>
          <p class="success-3d-subtitle">
            All files transformed into <strong>{{ state.targetFormat().toUpperCase() }}</strong> with ultra-fast 3D processing.
          </p>
        </div>

        <!-- 3D Metric Podium -->
        <div class="metrics-3d-podium">
          <div class="metric-3d-pillar">
            <span class="metric-3d-label">Files Converted</span>
            <span class="metric-3d-value">{{ state.convertedImagesCount() }}</span>
          </div>
          <div class="podium-divider"></div>
          <div class="metric-3d-pillar">
            <span class="metric-3d-label">Original Total</span>
            <span class="metric-3d-value">{{ formatSize(state.totalOriginalSize()) }}</span>
          </div>
          <div class="podium-divider"></div>
          <div class="metric-3d-pillar">
            <span class="metric-3d-label">New 3D Size</span>
            <span class="metric-3d-value highlight-3d">{{ formatSize(state.totalConvertedSize()) }}</span>
          </div>
          @if (totalSavingsPercent() > 0) {
            <div class="podium-divider"></div>
            <div class="metric-3d-pillar">
              <span class="metric-3d-label">Total Saved</span>
              <span class="savings-badge-3d">-{{ totalSavingsPercent() }}%</span>
            </div>
          }
        </div>

        <!-- 3D Master Action Buttons -->
        <div class="primary-3d-actions">
          <button
            type="button"
            class="btn-3d-primary btn-master-download-3d"
            (click)="state.downloadAll()"
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            <span>
              {{ state.convertedImagesCount() > 1 ? 'Download All Converted (ZIP)' : 'Download Converted Image' }}
            </span>
          </button>

          <!-- Proceed to Remove Background Button -->
          <button
            type="button"
            class="btn-3d-accent btn-bg-remove-3d"
            (click)="proceedToBgRemover()"
            title="Send converted image directly to Background Remover Studio"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
            <span>Remove Background</span>
          </button>

          <button
            type="button"
            class="btn-3d-secondary btn-restart-3d"
            (click)="state.reset()"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 5v14M5 12h14"></path>
            </svg>
            <span>Convert More</span>
          </button>
        </div>

        <!-- 3D Converted Items Card -->
        <div class="converted-3d-card">
          <div class="list-3d-header">
            <span class="list-3d-title">Converted Files ({{ state.convertedImagesCount() }})</span>
            <button type="button" class="link-3d-adjust" (click)="state.goToWorkspace()">
              Adjust 3D Settings
            </button>
          </div>

          <div class="items-3d-track">
            @for (item of state.images(); track item.id) {
              @if (item.status === 'done') {
                <div class="converted-3d-item">
                  <div class="item-3d-preview">
                    @if (item.convertedUrl || item.previewUrl) {
                      <img [src]="item.convertedUrl || item.previewUrl" [alt]="item.name" />
                    }
                  </div>

                  <div class="item-3d-meta-col">
                    <div class="name-3d-row">
                      <span class="item-orig-name" [title]="item.name">{{ item.name }}</span>
                      <span class="arrow-3d">➔</span>
                      <span class="item-conv-name">{{ item.convertedName }}</span>
                    </div>
                    <div class="size-3d-row">
                      <span class="old-size-tag">{{ formatSize(item.originalSize) }}</span>
                      <span class="arrow-3d">➔</span>
                      <span class="new-size-tag">{{ formatSize(item.convertedSize || 0) }}</span>
                      @if (calcSavings(item) > 0) {
                        <span class="savings-3d-pill">-{{ calcSavings(item) }}%</span>
                      }
                    </div>
                  </div>

                  <div class="item-action-btns">
                    <button
                      type="button"
                      class="btn-3d-accent-sm btn-item-bg-3d"
                      (click)="proceedToBgRemover(item)"
                      title="Remove background from this image"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                      </svg>
                      <span>Remove BG</span>
                    </button>

                    <button
                      type="button"
                      class="btn-3d-primary btn-single-dl-3d"
                      (click)="state.downloadSingle(item.id)"
                      title="Download this file"
                    >
                      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                      </svg>
                      <span>Download</span>
                    </button>
                  </div>
                </div>
              }
            }
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .result-3d-wrapper {
      min-height: calc(100vh - 120px);
      padding: 44px 20px 80px;
      display: flex;
      justify-content: center;
      background-color: var(--bg-page);
      perspective: 1200px;
      transition: background-color 0.3s ease;
    }

    .result-3d-container {
      width: 100%;
      max-width: 800px;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      animation: fadeIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* 3D Success Header */
    .success-3d-header {
      margin-bottom: 30px;
    }

    .trophy-badge-3d {
      width: 72px;
      height: 72px;
      border-radius: var(--radius-lg);
      background: var(--gold-gradient);
      color: #ffffff;
      border: 2px solid rgba(255, 255, 255, 0.4);
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 18px;
      box-shadow: 0 8px 0 #b45309, 0 16px 30px rgba(245, 158, 11, 0.4);
      position: relative;
      transform: rotateY(-8deg) rotateX(6deg);
      transition: transform 0.3s ease;
    }

    .trophy-badge-3d:hover {
      transform: rotateY(0deg) rotateX(0deg) scale(1.08);
    }

    .trophy-glow-ring {
      position: absolute;
      inset: -6px;
      border-radius: var(--radius-lg);
      background: radial-gradient(circle, rgba(245, 158, 11, 0.4), transparent 70%);
      filter: blur(8px);
      z-index: -1;
      animation: pulseGlow 2s infinite ease-in-out;
    }

    .success-3d-title {
      font-size: 36px;
      font-weight: 900;
      color: var(--text-main);
      margin-bottom: 8px;
      letter-spacing: -0.6px;
    }

    .success-3d-subtitle {
      font-size: 15px;
      color: var(--text-muted);
    }

    .success-3d-subtitle strong {
      color: var(--text-main);
    }

    /* 3D Metric Podium */
    .metrics-3d-podium {
      width: 100%;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 20px 28px;
      display: flex;
      align-items: center;
      justify-content: space-around;
      box-shadow: var(--3d-card-shadow);
      margin-bottom: 32px;
      flex-wrap: wrap;
      gap: 14px;
      transition: all 0.3s ease;
    }

    .metric-3d-pillar {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 3px;
    }

    .metric-3d-label {
      font-size: 11px;
      font-weight: 800;
      color: var(--text-light);
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }

    .metric-3d-value {
      font-size: 18px;
      font-weight: 900;
      color: var(--text-main);
    }

    .metric-3d-value.highlight-3d {
      color: var(--primary);
    }

    .savings-badge-3d {
      background: var(--success-bg);
      color: var(--success-text);
      border: 1px solid var(--success-border);
      font-size: 14px;
      font-weight: 900;
      padding: 3px 10px;
      border-radius: var(--radius-pill);
      box-shadow: 0 2px 0 var(--success-border);
    }

    .podium-divider {
      width: 1px;
      height: 32px;
      background: var(--border-subtle);
    }

    /* 3D Primary Actions */
    .primary-3d-actions {
      display: flex;
      gap: 14px;
      width: 100%;
      margin-bottom: 36px;
      flex-wrap: wrap;
    }

    .btn-master-download-3d {
      flex: 1.2;
      min-width: 240px;
      font-size: 16px;
      padding: 16px 28px;
      gap: 10px;
    }

    .btn-bg-remove-3d {
      flex: 1;
      min-width: 210px;
      font-size: 16px;
      padding: 16px 24px;
      gap: 10px;
    }

    .btn-3d-accent {
      background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
      color: #ffffff;
      border: 1px solid rgba(255, 255, 255, 0.25);
      border-radius: var(--radius-md);
      font-family: inherit;
      font-weight: 800;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: 0 5px 0 #4f46e5, 0 12px 24px rgba(99, 102, 241, 0.35);
      transform: translateY(0);
      transition: all 0.15s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .btn-3d-accent:hover {
      transform: translateY(-2px);
      box-shadow: 0 7px 0 #4f46e5, 0 16px 28px rgba(99, 102, 241, 0.45);
      background: linear-gradient(135deg, #9333ea 0%, #4f46e5 100%);
    }

    .btn-3d-accent:active {
      transform: translateY(3px);
      box-shadow: 0 2px 0 #4f46e5, 0 4px 10px rgba(99, 102, 241, 0.2);
    }

    .btn-3d-accent-sm {
      background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
      color: #ffffff;
      border: 1px solid rgba(255, 255, 255, 0.25);
      border-radius: var(--radius-sm);
      font-family: inherit;
      font-weight: 800;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: 0 3px 0 #4f46e5;
      transform: translateY(0);
      transition: all 0.15s ease;
    }

    .btn-3d-accent-sm:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 0 #4f46e5, 0 6px 12px rgba(99, 102, 241, 0.25);
    }

    .btn-3d-accent-sm:active {
      transform: translateY(2px);
      box-shadow: 0 1px 0 #4f46e5;
    }

    .btn-item-bg-3d {
      padding: 9px 14px;
      font-size: 12px;
      gap: 6px;
      flex-shrink: 0;
    }

    .btn-restart-3d {
      font-size: 15px;
      padding: 16px 22px;
      gap: 8px;
    }

    /* 3D Converted List Card */
    .converted-3d-card {
      width: 100%;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-xl);
      padding: 24px;
      box-shadow: var(--3d-card-shadow);
      text-align: left;
      transition: background-color 0.3s ease, border-color 0.3s ease;
    }

    .list-3d-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border-subtle);
    }

    .list-3d-title {
      font-size: 15px;
      font-weight: 900;
      color: var(--text-main);
    }

    .link-3d-adjust {
      font-size: 12px;
      font-weight: 800;
      color: var(--primary);
    }

    .link-3d-adjust:hover {
      text-decoration: underline;
    }

    .items-3d-track {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .converted-3d-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 16px;
      background: var(--bg-subtle);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      box-shadow: 0 3px 0 var(--border-hover);
      gap: 16px;
      transform: translateY(0);
      transition: all 0.15s ease;
    }

    .converted-3d-item:hover {
      background: var(--bg-card-hover);
      transform: translateY(-2px);
      box-shadow: 0 5px 0 var(--border-hover), 0 8px 16px rgba(0, 0, 0, 0.04);
    }

    .item-3d-preview {
      width: 48px;
      height: 48px;
      border-radius: var(--radius-sm);
      overflow: hidden;
      background-color: var(--bg-subtle);
      background-image: 
        linear-gradient(45deg, var(--checker-color) 25%, transparent 25%),
        linear-gradient(-45deg, var(--checker-color) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, var(--checker-color) 75%),
        linear-gradient(-45deg, transparent 75%, var(--checker-color) 75%);
      background-size: 10px 10px;
      background-position: 0 0, 0 5px, 5px -5px, -5px 0px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }

    .item-3d-preview img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    .item-3d-meta-col {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 0;
    }

    .name-3d-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      font-weight: 700;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .item-orig-name {
      color: var(--text-muted);
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .arrow-3d {
      color: var(--text-light);
      font-size: 11px;
    }

    .item-conv-name {
      color: var(--text-main);
      font-weight: 800;
    }

    .size-3d-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }

    .old-size-tag {
      color: var(--text-light);
      text-decoration: line-through;
    }

    .new-size-tag {
      color: var(--text-main);
      font-weight: 800;
    }

    .savings-3d-pill {
      background: var(--success-bg);
      color: var(--success-text);
      border: 1px solid var(--success-border);
      font-size: 11px;
      font-weight: 800;
      padding: 2px 7px;
      border-radius: 4px;
      box-shadow: 0 1px 0 var(--success-border);
    }

    .item-action-btns {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }

    .btn-single-dl-3d {
      padding: 9px 16px;
      font-size: 12px;
      gap: 6px;
      box-shadow: 0 3px 0 var(--3d-btn-primary-edge);
      flex-shrink: 0;
    }

    @media (max-width: 640px) {
      .success-3d-title {
        font-size: 28px;
      }
      .metrics-3d-podium {
        flex-direction: column;
        align-items: stretch;
      }
      .podium-divider {
        display: none;
      }
      .metric-3d-pillar {
        flex-direction: row;
        justify-content: space-between;
      }
      .converted-3d-item {
        flex-direction: column;
        align-items: stretch;
      }
      .item-action-btns {
        width: 100%;
        justify-content: space-between;
      }
      .btn-single-dl-3d, .btn-item-bg-3d {
        flex: 1;
        justify-content: center;
      }
    }
  `]
})
export class ResultScreenComponent {
  state = inject(ConversionStateService);
  bgRemover = inject(BgRemoverService);

  formatSize(bytes: number): string {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  calcSavings(item: ImageItem): number {
    if (!item.originalSize || !item.convertedSize) return 0;
    const diff = item.originalSize - item.convertedSize;
    if (diff <= 0) return 0;
    return Math.round((diff / item.originalSize) * 100);
  }

  totalSavingsPercent(): number {
    const orig = this.state.totalOriginalSize();
    const conv = this.state.totalConvertedSize();
    if (!orig || !conv || conv >= orig) return 0;
    return Math.round(((orig - conv) / orig) * 100);
  }

  proceedToBgRemover(item?: ImageItem): void {
    const targetItem = item || this.state.images().find((img) => img.status === 'done');
    if (!targetItem) return;

    const blob = targetItem.convertedBlob || targetItem.file;
    const fileName = targetItem.convertedName || targetItem.name;

    if (blob) {
      this.bgRemover.setImageFromBlob(blob, fileName);
    } else if (targetItem.file) {
      this.bgRemover.setImage(targetItem.file);
    }

    this.state.activeTool.set('bg-remover');
  }
}

