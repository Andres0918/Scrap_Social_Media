import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Card } from 'primeng/card';
import { Divider } from 'primeng/divider';
import { Badge } from 'primeng/badge';
import { Tag } from 'primeng/tag';
import { Accordion, AccordionContent, AccordionHeader, AccordionPanel } from 'primeng/accordion';
import { Storytelling } from '@/app/pages/dashboard/models/models';


@Component({
  selector: 'app-storytelling',
  standalone: true,
  imports: [
    CommonModule,
    Card,
    Divider,
    Badge,
    Tag,
    Accordion,
    AccordionPanel,
    AccordionHeader,
    AccordionContent,
  ],
  template: `
    <div class="storytelling-container" *ngIf="storytelling">
      <p-card styleClass="storytelling-card">
        <ng-template pTemplate="header">
          <div class="storytelling-header">
            <div class="header-icon">
              <i class="pi pi-book"></i>
            </div>
            <div class="header-text">
              <h2>📖 Narrativa de Resultados</h2>
              <p>Insights cualitativos del análisis</p>
            </div>
          </div>
        </ng-template>

        <!-- Overall Sentiment -->
        <div class="overall-sentiment">
          <div class="sentiment-badge" [class]="(storytelling.overall_sentiment || 'unknown').toLowerCase()">
            <i [class]="getSentimentIcon(storytelling.overall_sentiment)"></i>
            <span>Sentimiento General: <strong>{{ storytelling.overall_sentiment }}</strong></span>
          </div>
        </div>

        <p-divider></p-divider>

        <!-- Key Insights -->
        <div class="insights-section">
          <h3 class="section-title">
            <i class="pi pi-lightbulb"></i>
            Descubrimientos Clave
          </h3>
          <div class="insights-grid">
            <div
              *ngFor="let insight of storytelling.key_insights; let i = index"
              class="insight-card"
              [class.slide-in-left]="i % 2 === 0"
              [class.slide-in-right]="i % 2 === 1"
            >
              <div class="insight-icon">
                <i [class]="getInsightIcon(insight.type)"></i>
              </div>
              <div class="insight-content">
                <div class="insight-type">{{ formatInsightType(insight.type) }}</div>
                <div class="insight-message">{{ insight.message }}</div>
              </div>
            </div>
          </div>
        </div>

        <p-divider></p-divider>

        <!-- Platform Comparison - PrimeNG 20 Syntax -->
        <div class="comparison-section" *ngIf="storytelling.platform_comparison">
          <h3 class="section-title">
            <i class="pi pi-chart-line"></i>
            Comparación por Plataforma
          </h3>
          <div class="platforms-comparison">
            <p-accordion [multiple]="true">
              <p-accordionpanel *ngFor="let platform of getPlatforms(); let idx = index" [value]="idx.toString()">
                <p-accordionheader>
                  <div class="accordion-header">
                    <i [class]="'pi ' + getPlatformIcon(platform)"></i>
                    <span>{{ formatPlatformName(platform) }}</span>
                    <p-badge [value]="getPlatformData(platform).total_comments.toString()"></p-badge>
                  </div>
                </p-accordionheader>

                <p-accordioncontent>
                  <div class="platform-details">
                    <div class="detail-row">
                      <div class="detail-label">
                        <i class="pi pi-heart-fill"></i>
                        Sentimiento Dominante
                      </div>
                      <div class="detail-value">
                        <p-tag
                          [value]="getPlatformData(platform).dominant_sentiment"
                          [severity]="getSentimentSeverity(getPlatformData(platform).dominant_sentiment)"
                        ></p-tag>
                      </div>
                    </div>

                    <div class="detail-row">
                      <div class="detail-label">
                        <i class="pi pi-comments"></i>
                        Total de Comentarios
                      </div>
                      <div class="detail-value">
                        {{ getPlatformData(platform).total_comments | number }}
                      </div>
                    </div>

                    <div class="detail-row">
                      <div class="detail-label">
                        <i class="pi pi-align-left"></i>
                        Longitud Promedio
                      </div>
                      <div class="detail-value">
                        {{ getPlatformData(platform).avg_length | number:'1.1-1' }} palabras
                      </div>
                    </div>

                    <!-- Sentiment Distribution -->
                    <div class="sentiment-distribution">
                      <div class="distribution-title">Distribución de Sentimientos</div>
                      <div class="distribution-bars">
                        <div
                          *ngFor="let sent of getSentiments(platform)"
                          class="dist-bar"
                          [style.width.%]="getPercentage(platform, sent)"
                          [class]="sent.toLowerCase()"
                        >
                          <span *ngIf="getPercentage(platform, sent) > 10">
                            {{ getPercentage(platform, sent) }}%
                          </span>
                        </div>
                      </div>
                      <div class="distribution-legend">
                        <span *ngFor="let sent of getSentiments(platform)" class="legend-item" [class]="sent.toLowerCase()">
                          <i class="pi pi-circle-fill"></i>
                          {{ sent }}: {{ getPlatformData(platform).distribution[sent] }}
                        </span>
                      </div>
                    </div>
                  </div>
                </p-accordioncontent>
              </p-accordionpanel>
            </p-accordion>
          </div>
        </div>
      </p-card>
    </div>
  `,
  styles: [`
    .storytelling-container {
      margin: 2rem 0;
    }

    ::ng-deep .storytelling-card {
      background: linear-gradient(135deg, #F093FB 0%, #F5576C 100%);
      border-radius: 24px !important;
      overflow: hidden;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);

      .p-card-body {
        background: white;
        padding: 2rem;
      }
    }

    .storytelling-header {
      background: linear-gradient(135deg, #F093FB 0%, #F5576C 100%);
      color: white;
      padding: 3rem 2rem;
      display: flex;
      align-items: center;
      gap: 2rem;

      .header-icon {
        background: rgba(255, 255, 255, 0.2);
        width: 80px;
        height: 80px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        backdrop-filter: blur(10px);
      }

      .header-text {
        h2 {
          margin: 0 0 0.5rem 0;
          font-size: 2.5rem;
          font-weight: 800;
        }

        p {
          margin: 0;
          opacity: 0.95;
          font-size: 1.2rem;
        }
      }
    }

    /* Overall Sentiment */
    .overall-sentiment {
      margin-bottom: 2rem;

      .sentiment-badge {
        display: inline-flex;
        align-items: center;
        gap: 1rem;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        font-size: 1.3rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);

        &.positivo { background: linear-gradient(135deg, #10B981 0%, #059669 100%); }
        &.negativo { background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); }
        &.neutro { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); }

        i {
          font-size: 2rem;
        }

        strong {
          font-weight: 800;
        }
      }
    }

    /* Section Title */
    .section-title {
      display: flex;
      align-items: center;
      gap: 1rem;
      font-size: 1.8rem;
      font-weight: 800;
      color: #1F2937;
      margin-bottom: 1.5rem;

      i {
        color: #F5576C;
        font-size: 2rem;
      }
    }

    /* Insights */
    .insights-grid {
      display: grid;
      gap: 1.5rem;
    }

    .insight-card {
      background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
      padding: 1.5rem;
      border-radius: 16px;
      display: flex;
      gap: 1.5rem;
      border-left: 5px solid #F5576C;
      transition: all 0.3s;

      &:hover {
        transform: translateX(10px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
      }

      .insight-icon {
        background: linear-gradient(135deg, #F093FB 0%, #F5576C 100%);
        color: white;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        flex-shrink: 0;
        box-shadow: 0 5px 15px rgba(245, 87, 108, 0.3);
      }

      .insight-content {
        flex: 1;

        .insight-type {
          font-size: 0.9rem;
          font-weight: 700;
          color: #F5576C;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 0.5rem;
        }

        .insight-message {
          font-size: 1.1rem;
          line-height: 1.6;
          color: #1F2937;
          font-weight: 500;
        }
      }
    }

    /* Platform Comparison - PrimeNG 20 Accordion Styles */
    ::ng-deep .p-accordion {
      .p-accordionpanel {
        margin-bottom: 0.5rem;

        .p-accordionheader {
          .p-accordionheader-toggle {
            background: #F9FAFB;
            border: none;
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.3s;

            &:hover {
              background: #F3F4F6;
            }

            .accordion-header {
              display: flex;
              align-items: center;
              gap: 1rem;
              font-size: 1.2rem;
              font-weight: 700;
              color: #1F2937;

              i {
                font-size: 1.5rem;
                color: #F5576C;
              }
            }
          }

          &.p-accordionheader-active .p-accordionheader-toggle {
            background: linear-gradient(135deg, #F093FB 0%, #F5576C 100%);
            color: white;

            .accordion-header {
              color: white;

              i {
                color: white;
              }
            }
          }
        }

        .p-accordioncontent {
          border: none;
          padding: 1.5rem;
          background: white;
        }
      }
    }

    .platform-details {
      display: flex;
      flex-direction: column;
      gap: 1.5rem;

      .detail-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background: #F9FAFB;
        border-radius: 12px;

        .detail-label {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          font-weight: 600;
          color: #374151;

          i {
            color: #F5576C;
            font-size: 1.2rem;
          }
        }

        .detail-value {
          font-weight: 700;
          color: #1F2937;
          font-size: 1.1rem;
        }
      }
    }

    /* Sentiment Distribution */
    .sentiment-distribution {
      background: #F9FAFB;
      padding: 1.5rem;
      border-radius: 12px;

      .distribution-title {
        font-weight: 700;
        color: #374151;
        margin-bottom: 1rem;
        font-size: 1.05rem;
      }

      .distribution-bars {
        display: flex;
        height: 40px;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 1rem;

        .dist-bar {
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-weight: 700;
          font-size: 0.9rem;
          transition: all 0.3s;

          &:hover {
            filter: brightness(1.1);
          }

          &.positivo { background: #10B981; }
          &.negativo { background: #EF4444; }
          &.neutro { background: #F59E0B; }
        }
      }

      .distribution-legend {
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;

        .legend-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.9rem;
          font-weight: 600;

          &.positivo { color: #10B981; }
          &.negativo { color: #EF4444; }
          &.neutro { color: #F59E0B; }

          i {
            font-size: 0.6rem;
          }
        }
      }
    }

    @media (max-width: 768px) {
      .storytelling-header {
        flex-direction: column;
        text-align: center;

        .header-text h2 {
          font-size: 2rem;
        }
      }

      .insight-card {
        flex-direction: column;
        text-align: center;
      }

      .platform-details .detail-row {
        flex-direction: column;
        gap: 0.5rem;
        text-align: center;
      }
    }
  `]
})
export class StorytellingComponent {
  @Input() storytelling!: Storytelling;

  getPlatforms(): string[] {
    return Object.keys(this.storytelling.platform_comparison || {});
  }

  getPlatformData(platform: string): any {
    return this.storytelling.platform_comparison[platform];
  }

  getSentiments(platform: string): string[] {
    return Object.keys(this.getPlatformData(platform).distribution);
  }

  getPercentage(platform: string, sentiment: string): number {
    const data = this.getPlatformData(platform);
    const total = data.total_comments;
    const count = data.distribution[sentiment];
    return Math.round((count / total) * 100);
  }

  formatPlatformName(platform: string): string {
    const names: { [key: string]: string } = {
      'linkedin': 'LinkedIn',
      'facebook': 'Facebook',
      'x': 'X/Twitter',
      'instagram': 'Instagram',
      'reddit': 'Reddit',
      'twitter': 'Twitter',
      'youtube': 'YouTube'
    };
    return names[platform] || platform;
  }

  getPlatformIcon(platform: string): string {
    const icons: { [key: string]: string } = {
      'linkedin': 'pi-linkedin',
      'facebook': 'pi-facebook',
      'x': 'pi-twitter',
      'instagram': 'pi-instagram',
      'reddit': 'pi-reddit',
      'twitter': 'pi-twitter',
      'youtube': 'pi-youtube'
    };
    return icons[platform] || 'pi-globe';
  }

  formatInsightType(type: string): string {
    return type.replace(/_/g, ' ').toUpperCase();
  }

  getInsightIcon(type: string): string {
    const icons: { [key: string]: string } = {
      'positive_dominance': 'pi-thumbs-up',
      'negative_alert': 'pi-exclamation-triangle',
      'platform_engagement': 'pi-chart-line',
      'trend': 'pi-arrow-up'
    };
    return icons[type] || 'pi-info-circle';
  }

  getSentimentIcon(sentiment: string): string {
    const icons: { [key: string]: string } = {
      'Positivo': 'pi pi-thumbs-up',
      'Negativo': 'pi pi-thumbs-down',
      'Neutro': 'pi pi-minus'
    };
    return icons[sentiment] || 'pi-question';
  }

  getSentimentSeverity(sentiment: string): 'success' | 'danger' | 'warn' {
    const severities: { [key: string]: any } = {
      'Positivo': 'success',
      'Negativo': 'danger',
      'Neutro': 'warn'
    };
    return severities[sentiment] || 'warn';
  }
}
