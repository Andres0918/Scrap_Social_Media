import { Component, Input } from '@angular/core';
import { Card } from 'primeng/card';
import { Timeline } from 'primeng/timeline';
import { WebSocketMessage } from '@/app/pages/dashboard/models/models';
import { Chip } from 'primeng/chip';
import { Tag } from 'primeng/tag';

@Component({
    selector: 'app-progress-tracker',
    template: `
    <div class="progress-container">
      <p-card styleClass="progress-card">
        <ng-template pTemplate="header">
          <div class="progress-header">
            <div class="header-icon pulse">
              <i class="pi pi-spin pi-spinner"></i>
            </div>
            <div class="header-text">
              <h3>Análisis en Progreso</h3>
              <p>Procesando comentarios en tiempo real...</p>
            </div>
          </div>
        </ng-template>

        <p-timeline [value]="messages" align="left">
          <ng-template pTemplate="marker" let-message>
            <div
              class="timeline-marker"
              [class.scraping]="message.status === 'scraping'"
              [class.processing]="message.status === 'nlp_processing'"
              [class.completed]="message.status === 'completed'"
              [class.error]="message.status === 'error'"
            >
              <i [class]="getStatusIcon(message.status)"></i>
            </div>
          </ng-template>

          <ng-template pTemplate="content" let-message>
            <div class="timeline-content">
              <div class="timeline-platform">
                <p-chip
                  *ngIf="message.platform"
                  [label]="formatPlatform(message.platform)"
                  [icon]="'pi ' + getPlatformIcon(message.platform)"
                  [styleClass]="'platform-chip ' + message.platform"
                ></p-chip>
                <p-tag
                  [value]="formatStatus(message.status)"
                  [severity]="getStatusSeverity(message.status)"
                ></p-tag>
              </div>
              <div class="timeline-message">
                {{ message.message || 'Procesando...' }}
              </div>
              <div class="timeline-details" *ngIf="message.progress || message.comments_found || message.total_analyzed">
                <div class="detail-item" *ngIf="message.progress">
                  <i class="pi pi-chart-line"></i>
                  Progreso: {{ message.progress }}
                </div>
                <div class="detail-item" *ngIf="message.comments_found">
                  <i class="pi pi-comments"></i>
                  Encontrados: {{ message.comments_found }}
                </div>
                <div class="detail-item" *ngIf="message.total_analyzed">
                  <i class="pi pi-check-circle"></i>
                  Analizados: {{ message.total_analyzed }}
                </div>
              </div>
            </div>
          </ng-template>
        </p-timeline>
      </p-card>
    </div>
  `,
    imports: [
        Card,
        Timeline,
        Chip,
        Tag
    ],
    styles: [`
    .progress-container {
      margin: 2rem 0;
    }

    ::ng-deep .progress-card {
      background: white;
      border-radius: 20px !important;
      overflow: hidden;

      .p-card-body {
        padding: 2rem;
      }
    }

    .progress-header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 2rem;
      display: flex;
      align-items: center;
      gap: 1.5rem;

      .header-icon {
        background: rgba(255, 255, 255, 0.2);
        width: 70px;
        height: 70px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        backdrop-filter: blur(10px);
      }

      .header-text {
        h3 {
          margin: 0 0 0.5rem 0;
          font-size: 1.8rem;
          font-weight: 800;
        }

        p {
          margin: 0;
          opacity: 0.9;
          font-size: 1.1rem;
        }
      }
    }

    /* Timeline */
    ::ng-deep .p-timeline {
      .p-timeline-event-marker {
        border: none;
        background: none;
      }

      .p-timeline-event-connector {
        background: #E5E7EB;
        width: 3px;
      }
    }

    .timeline-marker {
      width: 50px;
      height: 50px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 1.5rem;
      box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
      transition: all 0.3s;

      &.scraping {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
      }

      &.processing {
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
      }

      &.completed {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
      }

      &.error {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
      }
    }

    .timeline-content {
      background: #F9FAFB;
      padding: 1.5rem;
      border-radius: 16px;
      margin-left: 1rem;
      border-left: 4px solid #8B5CF6;
    }

    .timeline-platform {
      display: flex;
      gap: 1rem;
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }

    ::ng-deep .platform-chip {
      font-weight: 600 !important;

      &.linkedin { background: rgba(0, 119, 181, 0.2) !important; color: #0077B5 !important; }
      &.facebook { background: rgba(24, 119, 242, 0.2) !important; color: #1877F2 !important; }
      &.x { background: rgba(0, 0, 0, 0.2) !important; color: #000000 !important; }
      &.instagram { background: rgba(228, 64, 95, 0.2) !important; color: #E4405F !important; }
    }

    .timeline-message {
      font-size: 1.05rem;
      color: #1F2937;
      margin-bottom: 1rem;
      font-weight: 500;
    }

    .timeline-details {
      display: flex;
      gap: 1.5rem;
      flex-wrap: wrap;

      .detail-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #6B7280;
        font-size: 0.95rem;
        font-weight: 600;

        i {
          color: #8B5CF6;
          font-size: 1.1rem;
        }
      }
    }
  `]
})
export class ProgressTrackerComponent {
    @Input() messages: WebSocketMessage[] = [];

    getStatusIcon(status: string): string {
        const icons: {[key: string]: string} = {
            'scraping': 'pi pi-download',
            'nlp_processing': 'pi pi-sparkles',
            'completed': 'pi pi-check',
            'error': 'pi pi-times',
            'aggregating': 'pi pi-chart-bar',
            'finished': 'pi pi-flag'
        };
        return icons[status] || 'pi pi-circle';
    }

    getStatusSeverity(status: string): 'success' | 'info' | 'warn' | 'danger' {
        const severities: {[key: string]: any} = {
            'scraping': 'info',
            'nlp_processing': 'warn',
            'completed': 'success',
            'error': 'danger',
            'aggregating': 'info',
            'finished': 'success'
        };
        return severities[status] || 'info';
    }

    formatStatus(status: string): string {
        const formats: {[key: string]: string} = {
            'scraping': 'Scraping',
            'nlp_processing': 'Procesando IA',
            'completed': 'Completado',
            'error': 'Error',
            'aggregating': 'Agregando',
            'finished': 'Finalizado'
        };
        return formats[status] || status;
    }

    formatPlatform(platform: string): string {
        const names: {[key: string]: string} = {
            'linkedin': 'LinkedIn',
            'facebook': 'Facebook',
            'x': 'X',
            'instagram': 'Instagram'
        };
        return names[platform] || platform;
    }

    getPlatformIcon(platform: string): string {
        const icons: {[key: string]: string} = {
            'linkedin': 'pi-linkedin',
            'facebook': 'pi-facebook',
            'x': 'pi-twitter',
            'instagram': 'pi-instagram'
        };
        return icons[platform] || 'pi-globe';
    }
}
