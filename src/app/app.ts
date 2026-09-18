import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HeaderComponent } from './components/header/header.component';
import { LandingComponent } from './components/landing/landing.component';
import { WorkspaceComponent } from './components/workspace/workspace.component';
import { ResultScreenComponent } from './components/result-screen/result-screen.component';
import { FooterComponent } from './components/footer/footer.component';
import { BgRemoverComponent } from './components/bg-remover/bg-remover.component';
import { ConversionStateService } from './services/conversion-state.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    HeaderComponent,
    LandingComponent,
    WorkspaceComponent,
    ResultScreenComponent,
    BgRemoverComponent,
    FooterComponent,
  ],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  state = inject(ConversionStateService);
}
