import { Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Card } from 'primeng/card';
import { UIChart } from 'primeng/chart';
import { Statistics, SocialComment } from '@/app/pages/dashboard/models/models';

@Component({
  selector: 'app-sentiment-charts',
  template: `
    <div class="charts-container" *ngIf="statistics">
      <div class="charts-grid">
        <!-- Sentiment Distribution Doughnut -->
        <p-card styleClass="chart-card">
          <ng-template pTemplate="header">
            <div class="chart-header">
              <i class="pi pi-chart-pie"></i>
              <span>Distribución de Sentimientos</span>
            </div>
          </ng-template>
          <p-chart
            type="doughnut"
            [data]="sentimentChartData"
            [options]="doughnutOptions"
            [style]="{'height': '350px'}"
          ></p-chart>
        </p-card>

        <!-- Platform Comparison Bar Chart -->
        <p-card styleClass="chart-card">
          <ng-template pTemplate="header">
            <div class="chart-header">
              <i class="pi pi-chart-bar"></i>
              <span>Comentarios por Plataforma</span>
            </div>
          </ng-template>
          <p-chart
            type="bar"
            [data]="platformChartData"
            [options]="barOptions"
            [style]="{'height': '350px'}"
          ></p-chart>
        </p-card>
      </div>

      <!-- Platform Sentiment Breakdown -->
      <p-card styleClass="breakdown-card">
        <ng-template pTemplate="header">
          <div class="chart-header">
            <i class="pi pi-list"></i>
            <span>Desglose por Plataforma</span>
          </div>
        </ng-template>
        <div class="breakdown-grid">
          <div
            *ngFor="let platform of platforms"
            class="platform-breakdown"
          >
            <div class="platform-info">
              <div class="platform-name">
                <i [class]="'pi ' + getPlatformIcon(platform)"></i>
                {{ platform }}
              </div>
              <div class="platform-count">
                {{ getPlatformCount(platform) }} comentarios
              </div>
            </div>
            <div class="sentiment-bars">
              <div class="sentiment-bar positive"
                   [style.width.%]="getPlatformSentimentPercentage(platform, 'Positivo')">
                <span>{{ getPlatformSentimentPercentage(platform, 'Positivo') }}%</span>
              </div>
              <div class="sentiment-bar negative"
                   [style.width.%]="getPlatformSentimentPercentage(platform, 'Negativo')">
                <span>{{ getPlatformSentimentPercentage(platform, 'Negativo') }}%</span>
              </div>
              <div class="sentiment-bar neutral"
                   [style.width.%]="getPlatformSentimentPercentage(platform, 'Neutro')">
                <span>{{ getPlatformSentimentPercentage(platform, 'Neutro') }}%</span>
              </div>
            </div>
            <div class="sentiment-legend">
              <span class="legend-item positive">
                <i class="pi pi-circle-fill"></i> Positivo
              </span>
              <span class="legend-item negative">
                <i class="pi pi-circle-fill"></i> Negativo
              </span>
              <span class="legend-item neutral">
                <i class="pi pi-circle-fill"></i> Neutro
              </span>
            </div>
          </div>
        </div>
      </p-card>
    </div>
  `,
  imports: [
    CommonModule,
    Card,
    UIChart
  ],
  styles: [`
    .charts-container {
      display: flex;
      flex-direction: column;
      gap: 2rem;
    }

    .charts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
      gap: 2rem;
    }

    ::ng-deep .chart-card, ::ng-deep .breakdown-card {
      background: white;
      border-radius: 20px !important;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
      overflow: hidden;

      .p-card-body {
        padding: 2rem;
      }
    }

    .chart-header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 1.5rem 2rem;
      display: flex;
      align-items: center;
      gap: 1rem;
      font-size: 1.3rem;
      font-weight: 700;

      i {
        font-size: 1.5rem;
      }
    }

    /* Platform Breakdown */
    .breakdown-grid {
      display: flex;
      flex-direction: column;
      gap: 2rem;
    }

    .platform-breakdown {
      background: #F9FAFB;
      padding: 1.5rem;
      border-radius: 16px;
      border-left: 4px solid #8B5CF6;
    }

    .platform-info {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;

      .platform-name {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.2rem;
        font-weight: 700;
        color: #1F2937;

        i {
          color: #8B5CF6;
          font-size: 1.5rem;
        }
      }

      .platform-count {
        color: #6B7280;
        font-weight: 600;
      }
    }

    .sentiment-bars {
      display: flex;
      gap: 0.5rem;
      height: 40px;
      border-radius: 10px;
      overflow: hidden;
      background: #E5E7EB;
    }

    .sentiment-bar {
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

      &.positive { background: #10B981; }
      &.negative { background: #EF4444; }
      &.neutral { background: #F59E0B; }

      span {
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
      }
    }

    .sentiment-legend {
      display: flex;
      gap: 1.5rem;
      margin-top: 1rem;
      justify-content: center;

      .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        font-weight: 600;

        &.positive { color: #10B981; }
        &.negative { color: #EF4444; }
        &.neutral { color: #F59E0B; }

        i {
          font-size: 0.6rem;
        }
      }
    }

    @media (max-width: 768px) {
      .charts-grid {
        grid-template-columns: 1fr;
      }
    }
  `]
})
export class SentimentChartsComponent implements OnChanges {
  @Input() statistics!: Statistics;
  @Input() comments: SocialComment[] = [];

  sentimentChartData: any;
  platformChartData: any;
  platforms: string[] = [];

  doughnutOptions = {
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          usePointStyle: true,
          font: { size: 14, weight: 'bold' }
        }
      }
    },
    cutout: '60%'
  };

  barOptions = {
    plugins: {
      legend: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: { font: { size: 12, weight: 'bold' } }
      },
      x: {
        ticks: { font: { size: 12, weight: 'bold' } }
      }
    }
  };

  ngOnChanges(): void {
    if (this.statistics) {
      this.updateCharts();
    }
  }

  private updateCharts(): void {
    const dist = this.statistics.stats.sentiment_distribution;
    const platforms = this.statistics.stats.platform_breakdown;

    // Sentiment Doughnut Chart
    this.sentimentChartData = {
      labels: Object.keys(dist),
      datasets: [{
        data: Object.values(dist),
        backgroundColor: ['#10B981', '#EF4444', '#F59E0B'],
        borderWidth: 0,
        hoverOffset: 10
      }]
    };

    // Platform Bar Chart
    this.platforms = Object.keys(platforms);
    this.platformChartData = {
      labels: this.platforms.map(p => this.formatPlatformName(p)),
      datasets: [{
        data: Object.values(platforms),
        backgroundColor: [
          'rgba(139, 92, 246, 0.8)',
          'rgba(236, 72, 153, 0.8)',
          'rgba(59, 130, 246, 0.8)',
          'rgba(16, 185, 129, 0.8)'
        ],
        borderRadius: 8
      }]
    };
  }

  getPlatformCount(platform: string): number {
    return this.statistics.stats.platform_breakdown[platform] || 0;
  }

  getPlatformSentimentPercentage(platform: string, sentiment: string): number {
    if (!this.comments || this.comments.length === 0) return 0;

    // Filter comments for specific platform
    const platformComments = this.comments.filter(c =>
      c.origen.toLowerCase() === platform.toLowerCase()
    );

    if (platformComments.length === 0) return 0;

    // Count specific sentiment
    const sentimentCount = platformComments.filter(c =>
      c.sentimiento.toLowerCase() === sentiment.toLowerCase()
    ).length;

    return Math.round((sentimentCount / platformComments.length) * 100);
  }

  getPlatformIcon(platform: string): string {
    const icons: { [key: string]: string } = {
      'linkedin': 'pi-linkedin',
      'facebook': 'pi-facebook',
      'x': 'pi-twitter',
      'instagram': 'pi-instagram'
    };
    return icons[platform] || 'pi-globe';
  }

  formatPlatformName(platform: string): string {
    const names: { [key: string]: string } = {
      'linkedin': 'LinkedIn',
      'facebook': 'Facebook',
      'x': 'X/Twitter',
      'instagram': 'Instagram'
    };
    return names[platform] || platform;
  }
}
