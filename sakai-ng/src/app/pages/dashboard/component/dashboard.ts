import { Component, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MessageService } from 'primeng/api';
import { Subscription } from 'rxjs';

// PrimeNG Components
import { Card } from 'primeng/card';
import { Button } from 'primeng/button';
import { Toast } from 'primeng/toast';
import { InputText } from 'primeng/inputtext';

// Models
import {
    Platform,
    SearchQuery,
    Statistics,
    Storytelling,
    WebSocketMessage,
    SocialComment
} from '../models/models';

// Services


// Components

import { ProgressTrackerComponent } from '../../components/progress-tracker/progress-tracker.component';
import { CommentsTableComponent } from '@/app/pages/dashboard/component/commentstable.component';
import { StatsCardsComponent } from '@/app/pages/components/stats-cards/stats-cards.component';
import { SentimentChartsComponent } from '@/app/pages/components/sentimental-charts/sentimental-charts.component';
import { StorytellingComponent } from '@/app/pages/components/story-telling/story-telling.component';
import { ApiService } from '@/app/pages/dashboard/service/api.service';
import { WebsocketService } from '@/app/pages/dashboard/service/websocket.service';

@Component({
    selector: 'app-dashboard',
    standalone: true,
    templateUrl: './dashboard.component.html',
    styleUrls: ['./dashboard.component.scss'],
    imports: [
        CommonModule,              // ← AGREGADO para *ngIf, *ngFor
        FormsModule,
        Card,
        Button,
        Toast,
        InputText,                 // ← AGREGADO
        CommentsTableComponent,
        StatsCardsComponent,
        SentimentChartsComponent,
        StorytellingComponent,
        ProgressTrackerComponent
    ],
    providers: [MessageService]
})
export class DashboardComponent implements OnInit, OnDestroy {
    // ================================
    // STATE MANAGEMENT
    // ================================
    currentTaskId: string | null = null;
    isAnalyzing: boolean = false;
    showResults: boolean = false;

    // ================================
    // DATA
    // ================================
    searchQuery: string = '';
    selectedPlatforms: Platform[] = [];
    availablePlatforms: Platform[] = [
        { id: 'linkedin', name: 'LinkedIn', icon: 'pi-linkedin', color: '#0077B5', enabled: true },
        { id: 'facebook', name: 'Facebook', icon: 'pi-facebook', color: '#1877F2', enabled: true },
        { id: 'x', name: 'X (Twitter)', icon: 'pi-twitter', color: '#000000', enabled: true },
        { id: 'instagram', name: 'Instagram', icon: 'pi-instagram', color: '#E4405F', enabled: true }
    ];

    progressMessages: WebSocketMessage[] = [];
    comments: SocialComment[] = [];
    statistics: Statistics | null = null;
    storytelling: Storytelling | null = null;

    // ================================
    // SUBSCRIPTIONS
    // ================================
    private wsSubscription?: Subscription;

    constructor(
        private apiService: ApiService,
        private wsService: WebsocketService,
        private messageService: MessageService,
        private cdr: ChangeDetectorRef  // ← AGREGADO para evitar ExpressionChangedAfterItHasBeenCheckedError
    ) {
        this.selectedPlatforms = [...this.availablePlatforms];
    }

    ngOnInit(): void {
        this.connectWebSocket();
        this.checkBackendHealth();
    }

    ngOnDestroy(): void {
        this.wsSubscription?.unsubscribe();
        this.wsService.disconnect();
    }

    // ================================
    // WEBSOCKET
    // ================================
    private connectWebSocket(): void {
        this.wsSubscription = this.wsService.connect().subscribe({
            next: (message: WebSocketMessage) => {
                this.handleWebSocketMessage(message);
            },
            error: (error) => {
                console.error('WebSocket error:', error);
                this.showError('Error de conexión en tiempo real');
            }
        });
    }

    private handleWebSocketMessage(message: WebSocketMessage): void {
        if (message.task_id === this.currentTaskId) {
            this.progressMessages.push(message);

            switch (message.status) {
                case 'scraping':
                    this.showInfo(`Scraping en ${message.platform}...`);
                    break;
                case 'nlp_processing':
                    this.showInfo(`Procesando ${message.platform} con IA...`);
                    break;
                case 'completed':
                    this.showSuccess(`${message.platform} completado: ${message.total_analyzed} comentarios`);
                    break;
                case 'finished':
                    this.handleAnalysisComplete();
                    break;
                case 'error':
                    this.showError(`Error en ${message.platform}: ${message.error}`);
                    break;
            }

            // ← AGREGADO: Forzar detección de cambios para evitar errores
            this.cdr.detectChanges();
        }
    }

    // ================================
    // ANALYSIS FLOW
    // ================================
    startAnalysis(): void {
        if (!this.searchQuery.trim()) {
            this.showWarning('Por favor ingresa un término de búsqueda');
            return;
        }

        if (this.selectedPlatforms.length === 0) {
            this.showWarning('Selecciona al menos una plataforma');
            return;
        }

        const query: SearchQuery = {
            query: this.searchQuery.trim(),
            platforms: this.selectedPlatforms.map(p => p.id)
        };

        this.isAnalyzing = true;
        this.showResults = false;
        this.progressMessages = [];

        this.apiService.startScraping(query).subscribe({
            next: (response) => {
                this.currentTaskId = response.task_id!;
                this.showSuccess('Análisis iniciado correctamente');
            },
            error: (error) => {
                console.error('Error starting analysis:', error);
                this.showError('Error al iniciar el análisis');
                this.isAnalyzing = false;
            }
        });
    }

    private handleAnalysisComplete(): void {
        this.showSuccess('¡Análisis completado!');
        this.loadResults();
    }

    private loadResults(): void {
        if (!this.currentTaskId) return;

        // Cargar estadísticas
        this.apiService.getStatistics(this.currentTaskId).subscribe({
            next: (stats) => {
                this.statistics = stats;
                this.cdr.detectChanges();  // ← AGREGADO
            },
            error: (error) => {
                console.error('Error loading statistics:', error);
            }
        });

        // Cargar comentarios
        this.apiService.getResults(this.currentTaskId).subscribe({
            next: (response) => {
                this.comments = response.comments || response.data || [];
                this.showResults = true;
                this.isAnalyzing = false;
                this.cdr.detectChanges();  // ← AGREGADO
            },
            error: (error) => {
                console.error('Error loading results:', error);
                this.showError('Error al cargar resultados');
                this.isAnalyzing = false;
            }
        });

        // Cargar storytelling
        this.apiService.getStorytelling(this.currentTaskId).subscribe({
            next: (story) => {
                this.storytelling = story;
                this.cdr.detectChanges();  // ← AGREGADO
            },
            error: (error) => {
                console.error('Error loading storytelling:', error);
            }
        });
    }

    // ================================
    // PLATFORM SELECTION
    // ================================
    togglePlatform(platform: Platform): void {
        const index = this.selectedPlatforms.findIndex(p => p.id === platform.id);
        if (index > -1) {
            this.selectedPlatforms.splice(index, 1);
        } else {
            this.selectedPlatforms.push(platform);
        }
    }

    isPlatformSelected(platform: Platform): boolean {
        return this.selectedPlatforms.some(p => p.id === platform.id);
    }

    // ================================
    // UTILITY
    // ================================
    private checkBackendHealth(): void {
        this.apiService.healthCheck().subscribe({
            next: () => {
                console.log('✅ Backend conectado');
            },
            error: (error) => {
                console.error('❌ Backend no disponible:', error);
                this.showError('No se puede conectar al servidor. Verifica que el backend esté corriendo en http://localhost:8000');
            }
        });
    }

    // ================================
    // NOTIFICATIONS
    // ================================
    private showSuccess(message: string): void {
        this.messageService.add({
            severity: 'success',
            summary: 'Éxito',
            detail: message,
            life: 3000
        });
    }

    private showError(message: string): void {
        this.messageService.add({
            severity: 'error',
            summary: 'Error',
            detail: message,
            life: 5000
        });
    }

    private showWarning(message: string): void {
        this.messageService.add({
            severity: 'warn',
            summary: 'Advertencia',
            detail: message,
            life: 3000
        });
    }

    private showInfo(message: string): void {
        this.messageService.add({
            severity: 'info',
            summary: 'Información',
            detail: message,
            life: 2000
        });
    }

    // ================================
    // METRICS HELPERS
    // ================================
    getPositivePercentage(): number {
        if (!this.statistics?.stats.sentiment_distribution) return 0;

        const total = this.statistics.stats.total_comments;
        const positive = this.statistics.stats.sentiment_distribution['Positivo'] ||
            this.statistics.stats.sentiment_distribution['positivo'] || 0;

        return total > 0 ? Math.round((positive / total) * 100) : 0;
    }

    getPlatformCount(): number {
        if (!this.statistics?.stats.platform_breakdown) return 0;
        return Object.keys(this.statistics.stats.platform_breakdown).length;
    }
}
