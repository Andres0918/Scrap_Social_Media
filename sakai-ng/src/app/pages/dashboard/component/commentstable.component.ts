import { Component, Input, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Card } from 'primeng/card';
import { TableModule } from 'primeng/table';
import { Chip } from 'primeng/chip';
import { Tag } from 'primeng/tag';
import { Badge } from 'primeng/badge';
import { InputText } from 'primeng/inputtext';
import { Tab, TabList, TabPanel, TabPanels, Tabs } from 'primeng/tabs';
import { SocialComment } from '@/app/pages/dashboard/models/models';


@Component({
    selector: 'app-comments-table',
    standalone: true,
    imports: [
        CommonModule,
        Card,
        TableModule,
        Chip,
        Tag,
        Badge,
        InputText,
        Tabs,
        TabList,
        Tab,
        TabPanels,
        TabPanel,
    ],
    template: `
    <div class="comments-container" *ngIf="comments && comments.length > 0">
      <p-card styleClass="comments-card">
        <ng-template pTemplate="header">
          <div class="comments-header">
            <div class="header-left">
              <i class="pi pi-comments"></i>
              <span>Comentarios Analizados</span>
              <p-badge [value]="comments.length.toString()"></p-badge>
            </div>
            <div class="header-right">
              <span class="p-input-icon-left">
                <i class="pi pi-search"></i>
                <input
                  pInputText
                  type="text"
                  (input)="applyFilter($event)"
                  placeholder="Buscar comentarios..."
                />
              </span>
            </div>
          </div>
        </ng-template>

        <!-- Platform Tabs - PrimeNG 20 Syntax -->
        <p-tabs [(value)]="activeTab">
          <p-tablist>
            <p-tab value="todos">
              <i class="pi pi-th-large"></i>
              Todos
            </p-tab>
            <p-tab *ngFor="let platform of platforms" [value]="platform">
              <i [class]="'pi ' + getPlatformIcon(platform)"></i>
              {{ formatPlatformName(platform) }}
            </p-tab>
          </p-tablist>

          <p-tabpanels>
            <p-tabpanel value="todos">
              <ng-container *ngTemplateOutlet="tableTemplate; context: {data: getFilteredComments()}"></ng-container>
            </p-tabpanel>
            <p-tabpanel *ngFor="let platform of platforms" [value]="platform">
              <ng-container *ngTemplateOutlet="tableTemplate; context: {data: getFilteredComments(platform)}"></ng-container>
            </p-tabpanel>
          </p-tabpanels>
        </p-tabs>
      </p-card>
    </div>

    <!-- Table Template -->
    <ng-template #tableTemplate let-data="data">
      <p-table
        [value]="data"
        [paginator]="true"
        [rows]="10"
        [rowsPerPageOptions]="[5, 10, 20, 50]"
        [tableStyle]="{'min-width': '50rem'}"
        styleClass="p-datatable-sm"
        [globalFilterFields]="['texto_original', 'sentimiento', 'explicacion']"
      >
        <ng-template pTemplate="header">
          <tr>
            <th pSortableColumn="origen" style="width: 120px">
              Plataforma <p-sortIcon field="origen"></p-sortIcon>
            </th>
            <th pSortableColumn="texto_original" style="width: 40%">
              Comentario <p-sortIcon field="texto_original"></p-sortIcon>
            </th>
            <th pSortableColumn="sentimiento" style="width: 150px">
              Sentimiento <p-sortIcon field="sentimiento"></p-sortIcon>
            </th>
            <th pSortableColumn="modelo" style="width: 150px">
              Modelo <p-sortIcon field="modelo"></p-sortIcon>
            </th>
            <th pSortableColumn="tiempo_ejecucion" style="width: 120px">
              Tiempo <p-sortIcon field="tiempo_ejecucion"></p-sortIcon>
            </th>
          </tr>
        </ng-template>
        <ng-template pTemplate="body" let-comment>
          <tr class="comment-row">
            <td>
              <p-chip
                [label]="formatPlatformName(comment.origen)"
                [icon]="'pi ' + getPlatformIcon(comment.origen)"
                [styleClass]="'platform-chip ' + comment.origen"
              ></p-chip>
            </td>
            <td>
              <div class="comment-text">
                {{ comment.texto_original }}
              </div>
              <div class="comment-explanation">
                <i class="pi pi-info-circle"></i>
                {{ comment.explicacion }}
              </div>
            </td>
            <td>
              <p-tag
                [value]="comment.sentimiento"
                [severity]="getSentimentSeverity(comment.sentimiento)"
                [icon]="getSentimentIcon(comment.sentimiento)"
              ></p-tag>
            </td>
            <td>
              <p-chip
                [label]="comment.modelo"
                icon="pi pi-sparkles"
                styleClass="model-chip"
              ></p-chip>
            </td>
            <td>
              <div class="time-badge">
                <i class="pi pi-clock"></i>
                {{ comment.tiempo_ejecucion }}s
              </div>
            </td>
          </tr>
        </ng-template>
        <ng-template pTemplate="emptymessage">
          <tr>
            <td colspan="5" class="empty-message">
              <i class="pi pi-inbox"></i>
              <p>No se encontraron comentarios</p>
            </td>
          </tr>
        </ng-template>
      </p-table>
    </ng-template>
  `,
    styles: [`
    .comments-container {
      margin-top: 2rem;
    }

    ::ng-deep .comments-card {
      background: white;
      border-radius: 20px !important;
      overflow: hidden;

      .p-card-body {
        padding: 0 !important;
      }
    }

    .comments-header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 1.5rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 2rem;

      .header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
        font-size: 1.3rem;
        font-weight: 700;

        i {
          font-size: 1.5rem;
        }

        ::ng-deep .p-badge {
          background: rgba(255, 255, 255, 0.3);
          color: white;
          font-size: 1rem;
          padding: 0.5rem 1rem;
        }
      }

      .header-right {
        ::ng-deep input {
          background: rgba(255, 255, 255, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.3);
          color: white;
          padding: 0.75rem 1rem 0.75rem 2.5rem;
          border-radius: 50px;
          font-size: 1rem;

          &::placeholder {
            color: rgba(255, 255, 255, 0.8);
          }

          &:focus {
            background: rgba(255, 255, 255, 0.3);
            border-color: rgba(255, 255, 255, 0.5);
          }
        }

        .p-input-icon-left > i {
          color: rgba(255, 255, 255, 0.8);
        }
      }
    }

    ::ng-deep .p-tabs {
      .p-tablist {
        background: #f8f9fa;
        border-bottom: 2px solid #e9ecef;

        .p-tab {
          font-weight: 600;
          color: #6c757d;
          padding: 1rem 1.5rem;
          transition: all 0.3s ease;

          i {
            margin-right: 0.5rem;
          }

          &:hover {
            color: #667eea;
            background: rgba(102, 126, 234, 0.1);
          }

          &.p-tab-active {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            background: white;
          }
        }
      }

      .p-tabpanels {
        padding: 1.5rem;
      }
    }

    ::ng-deep .p-datatable {
      .comment-row {
        transition: all 0.3s ease;

        &:hover {
          background: #f8f9fa !important;
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
      }

      .comment-text {
        font-size: 0.95rem;
        color: #2d3748;
        line-height: 1.6;
        margin-bottom: 0.5rem;
      }

      .comment-explanation {
        font-size: 0.85rem;
        color: #718096;
        display: flex;
        align-items: center;
        gap: 0.5rem;

        i {
          color: #667eea;
        }
      }

      .platform-chip {
        font-weight: 600;
        border-radius: 20px;

        &.reddit {
          background: linear-gradient(135deg, #FF4500, #FF6347) !important;
          color: white !important;
        }

        &.twitter {
          background: linear-gradient(135deg, #1DA1F2, #0d8bd9) !important;
          color: white !important;
        }

        &.youtube {
          background: linear-gradient(135deg, #FF0000, #cc0000) !important;
          color: white !important;
        }
      }

      .model-chip {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        font-weight: 600;
      }

      .time-badge {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        color: #667eea;

        i {
          font-size: 1rem;
        }
      }

      .empty-message {
        text-align: center;
        padding: 3rem;
        color: #718096;

        i {
          font-size: 3rem;
          color: #cbd5e0;
          display: block;
          margin-bottom: 1rem;
        }

        p {
          font-size: 1.1rem;
        }
      }
    }
  `]
})
export class CommentsTableComponent implements OnInit {
    @Input() comments: SocialComment[] = [];
    @Input() taskId: string = '';

    platforms: string[] = [];
    activeTab: string = 'todos';
    filteredComments: SocialComment[] = [];
    searchTerm: string = '';

    ngOnInit() {
        this.extractPlatforms();
        this.filteredComments = this.comments;
    }

    ngOnChanges() {
        this.extractPlatforms();
        this.applyCurrentFilter();
    }

    extractPlatforms() {
        const platformSet = new Set(this.comments.map(c => c.origen));
        this.platforms = Array.from(platformSet);
    }

    applyFilter(event: Event) {
        const input = event.target as HTMLInputElement;
        this.searchTerm = input.value.toLowerCase();
        this.applyCurrentFilter();
    }

    applyCurrentFilter() {
        let filtered = this.comments;

        if (this.searchTerm) {
            filtered = filtered.filter(comment =>
                comment.texto_original.toLowerCase().includes(this.searchTerm) ||
                comment.sentimiento.toLowerCase().includes(this.searchTerm) ||
                comment.explicacion.toLowerCase().includes(this.searchTerm)
            );
        }

        this.filteredComments = filtered;
    }

    getFilteredComments(platform?: string): SocialComment[] {
        if (!platform) {
            return this.filteredComments;
        }
        return this.filteredComments.filter(c => c.origen === platform);
    }

    formatPlatformName(platform: string): string {
        const names: { [key: string]: string } = {
            'reddit': 'Reddit',
            'twitter': 'Twitter/X',
            'youtube': 'YouTube'
        };
        return names[platform] || platform;
    }

    getPlatformIcon(platform: string): string {
        const icons: { [key: string]: string } = {
            'reddit': 'pi-reddit',
            'twitter': 'pi-twitter',
            'youtube': 'pi-youtube'
        };
        return icons[platform] || 'pi-globe';
    }

    getSentimentSeverity(sentiment: string): 'success' | 'danger' | 'warn' | 'info' {
        const severities: { [key: string]: 'success' | 'danger' | 'warn' | 'info' } = {
            'positivo': 'success',
            'negativo': 'danger',
            'neutro': 'warn'
        };
        return severities[sentiment.toLowerCase()] || 'info';
    }

    getSentimentIcon(sentiment: string): string {
        const icons: { [key: string]: string } = {
            'positivo': 'pi-thumbs-up',
            'negativo': 'pi-thumbs-down',
            'neutro': 'pi-minus'
        };
        return icons[sentiment.toLowerCase()] || 'pi-circle';
    }
}
