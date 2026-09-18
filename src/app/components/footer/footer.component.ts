import { Component } from '@angular/core';

@Component({
  selector: 'app-footer',
  standalone: true,
  template: `
    <footer class="app-footer-3d">
      <div class="footer-inner">
        <p class="copyright">
          © 2026 ImageConvert 3D • 100% Client-Side In-Browser Image Processing
        </p>
        <div class="footer-privacy-note-3d">
          <span class="privacy-lock-icon">🔒</span>
          <span>Zero files uploaded to any server. Your data stays private on your machine.</span>
        </div>
      </div>
    </footer>
  `,
  styles: [`
    .app-footer-3d {
      background-color: var(--bg-card);
      border-top: 1px solid var(--border-subtle);
      padding: 20px 24px;
      font-size: 12px;
      color: var(--text-muted);
      box-shadow: 0 -4px 16px rgba(0, 0, 0, 0.02);
      transition: background-color 0.3s ease, border-color 0.3s ease;
      position: relative;
      z-index: 10;
    }

    .footer-inner {
      max-width: 1240px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }

    .copyright {
      font-weight: 600;
    }

    .footer-privacy-note-3d {
      display: flex;
      align-items: center;
      gap: 6px;
      font-weight: 600;
      color: var(--text-muted);
    }

    .privacy-lock-icon {
      font-size: 14px;
    }

    @media (max-width: 640px) {
      .footer-inner {
        flex-direction: column;
        text-align: center;
        gap: 8px;
      }
    }
  `]
})
export class FooterComponent {}
