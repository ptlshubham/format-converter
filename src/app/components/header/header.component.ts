import { CommonModule } from '@angular/common';
import { Component, inject } from '@angular/core';
import { ConversionStateService } from '../../services/conversion-state.service';
import { ThemeService } from '../../services/theme.service';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule],
  template: `
    <header class="app-header-3d">
      <div class="header-container">
        <!-- 3D Modern Brand Logo -->
        <div class="brand-3d" (click)="onBrandClick()">
          <div class="brand-icon-3d">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="4" ry="4"></rect>
              <circle cx="8.5" cy="8.5" r="1.5"></circle>
              <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
          </div>
          <div class="brand-text">
            <span class="brand-title">Image<span class="highlight">Convert</span></span>
            <span class="brand-badge-3d">3D Studio</span>
          </div>
        </div>

        <!-- Tool Switcher Navigation -->
        <nav class="tool-nav-tabs">
          <button
            type="button"
            class="tool-tab-btn"
            [class.active]="state.activeTool() === 'converter'"
            (click)="state.activeTool.set('converter')"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
              <path d="M7 16V4M7 4L3 8M7 4L11 8M17 8V20M17 20L21 16M17 20L13 16"></path>
            </svg>
            Format Converter
          </button>
          <button
            type="button"
            class="tool-tab-btn"
            [class.active]="state.activeTool() === 'bg-remover'"
            (click)="state.activeTool.set('bg-remover')"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
            Background Remover
            <span class="ai-chip">In-House CV</span>
          </button>
        </nav>

        <!-- 3D Status / Theme Toggle / Actions -->
        <div class="header-actions">
          @if (state.activeTool() === 'converter') {
            <div class="privacy-pill-3d">
              <span class="live-pulse-node"></span>
              <span class="privacy-text">100% Client-Side</span>
            </div>
          } @else {
            <div class="privacy-pill-3d server-pill">
              <span class="live-pulse-node server-pulse"></span>
              <span class="privacy-text">Private In-Memory CV</span>
            </div>
          }

          <!-- 3D Tactile Sun / Moon Switcher -->
          <button
            type="button"
            class="theme-btn-3d"
            (click)="theme.toggleTheme()"
            [title]="theme.isDark() ? 'Switch to Light Mode' : 'Switch to Dark Mode'"
            aria-label="Toggle dark mode"
          >
            @if (theme.isDark()) {
              <svg class="theme-icon sun-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                <line x1="1" y1="12" x2="3" y2="12"></line>
                <line x1="21" y1="12" x2="23" y2="12"></line>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
              </svg>
            } @else {
              <svg class="theme-icon moon-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
              </svg>
            }
          </button>

          @if (state.step() !== 'landing') {
            <button
              type="button"
              class="btn-new-3d"
              (click)="state.reset()"
              title="Start New Conversion"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="M12 5v14M5 12h14"></path>
              </svg>
              <span class="btn-new-text">New</span>
            </button>
          }
        </div>
      </div>
    </header>
  `,
  styles: [`
    .app-header-3d {
      background: var(--bg-glass);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-subtle);
      position: sticky;
      top: 0;
      z-index: 50;
      height: 68px;
      display: flex;
      align-items: center;
      box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.06), 0 2px 0 var(--border-subtle);
      transition: background-color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .header-container {
      width: 100%;
      max-width: 1320px;
      margin: 0 auto;
      padding: 0 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }

    .brand-3d {
      display: flex;
      align-items: center;
      gap: 10px;
      cursor: pointer;
      user-select: none;
      perspective: 800px;
    }

    .brand-icon-3d {
      width: 38px;
      height: 38px;
      border-radius: var(--radius-md);
      background: var(--primary-gradient);
      color: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 0 var(--3d-btn-primary-edge), 0 8px 16px rgba(37, 99, 235, 0.35);
      transform: rotateY(-6deg) rotateX(4deg);
      transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      flex-shrink: 0;
    }

    .brand-3d:hover .brand-icon-3d {
      transform: rotateY(0deg) rotateX(0deg) scale(1.08) translateY(-2px);
      box-shadow: 0 6px 0 var(--3d-btn-primary-edge), 0 12px 24px rgba(37, 99, 235, 0.5);
    }

    .brand-text {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .brand-title {
      font-family: 'Outfit', sans-serif;
      font-size: clamp(17px, 2.5vw, 20px);
      font-weight: 800;
      color: var(--text-main);
      letter-spacing: -0.4px;
      white-space: nowrap;
    }

    .brand-title .highlight {
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-badge-3d {
      font-size: 11px;
      font-weight: 700;
      color: var(--primary);
      background: var(--primary-light);
      padding: 2px 8px;
      border-radius: var(--radius-pill);
      border: 1px solid var(--border-subtle);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4);
      white-space: nowrap;
    }

    .tool-nav-tabs {
      display: flex;
      align-items: center;
      gap: 6px;
      background: var(--bg-main);
      padding: 4px;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-subtle);
    }

    .tool-tab-btn {
      display: flex;
      align-items: center;
      gap: 7px;
      padding: 7px 14px;
      border-radius: var(--radius-sm);
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .tool-tab-btn:hover {
      color: var(--text-main);
    }

    .tool-tab-btn.active {
      background: var(--bg-card);
      color: var(--primary);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }

    .ai-chip {
      font-size: 10px;
      background: var(--primary-light);
      color: var(--primary);
      padding: 1px 6px;
      border-radius: var(--radius-pill);
      border: 1px solid var(--border-subtle);
    }

    .server-pill {
      color: #3b82f6 !important;
      background: rgba(59, 130, 246, 0.1) !important;
      border-color: rgba(59, 130, 246, 0.25) !important;
    }

    .server-pulse {
      background-color: #3b82f6 !important;
      box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .privacy-pill-3d {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 700;
      color: var(--success-text);
      background: var(--success-bg);
      border: 1px solid var(--success-border);
      padding: 6px 12px;
      border-radius: var(--radius-pill);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.3), 0 2px 4px rgba(0, 0, 0, 0.04);
      white-space: nowrap;
    }

    .live-pulse-node {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background-color: var(--success);
      box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.3);
      animation: pulseLive 2s infinite ease-in-out;
      flex-shrink: 0;
    }

    @keyframes pulseLive {
      0%, 100% {
        transform: scale(1);
        opacity: 1;
      }
      50% {
        transform: scale(1.35);
        opacity: 0.6;
      }
    }

    .theme-btn-3d {
      width: 38px;
      height: 38px;
      border-radius: var(--radius-md);
      background: var(--bg-card);
      color: var(--text-main);
      border: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 3px 0 var(--border-hover), 0 6px 12px rgba(0, 0, 0, 0.06);
      transform: translateY(0);
      transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
      flex-shrink: 0;
    }

    .theme-btn-3d:hover {
      transform: translateY(-2px);
      box-shadow: 0 5px 0 var(--border-hover), 0 10px 18px rgba(0, 0, 0, 0.1);
      border-color: var(--border-hover);
    }

    .theme-btn-3d:active {
      transform: translateY(3px);
      box-shadow: 0 0 0 var(--border-hover);
    }

    .sun-icon {
      color: #fbbf24;
      filter: drop-shadow(0 0 6px rgba(251, 191, 36, 0.5));
    }

    .moon-icon {
      color: #818cf8;
      filter: drop-shadow(0 0 6px rgba(129, 140, 248, 0.5));
    }

    .btn-new-3d {
      display: flex;
      align-items: center;
      gap: 5px;
      background: var(--bg-card);
      color: var(--text-main);
      border: 1px solid var(--border-subtle);
      padding: 7px 14px;
      border-radius: var(--radius-sm);
      font-size: 12px;
      font-weight: 700;
      box-shadow: 0 3px 0 var(--border-hover), 0 4px 10px rgba(0, 0, 0, 0.05);
      transform: translateY(0);
      transition: all 0.15s ease;
      white-space: nowrap;
    }

    .btn-new-3d:hover {
      transform: translateY(-2px);
      box-shadow: 0 5px 0 var(--border-hover), 0 8px 16px rgba(0, 0, 0, 0.08);
      border-color: var(--border-hover);
    }

    .btn-new-3d:active {
      transform: translateY(3px);
      box-shadow: 0 0 0 var(--border-hover);
    }

    @media (max-width: 768px) {
      .header-container {
        padding: 0 16px;
      }
      .brand-badge-3d {
        display: none;
      }
    }

    @media (max-width: 520px) {
      .privacy-text {
        display: none;
      }
      .privacy-pill-3d {
        padding: 6px 8px;
      }
    }
  `]
})
export class HeaderComponent {
  state = inject(ConversionStateService);
  theme = inject(ThemeService);

  onBrandClick(): void {
    this.state.reset();
  }
}
