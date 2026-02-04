import { Component, Input, OnChanges } from '@angular/core';
import { Statistics } from '@/app/pages/dashboard/models/models';
import { CommonModule } from '@angular/common';
import { Knob } from 'primeng/knob';
import { FormsModule } from '@angular/forms';
import { Card } from 'primeng/card';

@Component({
    selector: 'app-stats-cards',
    templateUrl: './stats-cards.component.html',
    imports: [
        CommonModule,
        Knob,
        FormsModule,
        Card
    ],
    styleUrls: ['./stats-cards.component.scss']
})
export class StatsCardsComponent implements OnChanges {
    @Input() statistics!: Statistics;

    positivePercentage: number = 0;
    negativePercentage: number = 0;
    neutralPercentage: number = 0;

    ngOnChanges(): void {
        if (this.statistics) {
            this.calculatePercentages();
        }
    }

    getDominantSentiment(): string {
        const dist = this.statistics.stats.sentiment_distribution;
        const entries = Object.entries(dist);
        if (entries.length === 0) return '-';

        const max = entries.reduce((a, b) => a[1] > b[1] ? a : b);
        return max[0];
    }

    getPlatformCount(): number {
        return Object.keys(this.statistics.stats.platform_breakdown).length;
    }

    private calculatePercentages(): void {
        const dist = this.statistics.stats.sentiment_distribution;
        const total = this.statistics.stats.total_comments || 1;

        this.positivePercentage = Math.round(((dist['Positivo'] || 0) / total) * 100);
        this.negativePercentage = Math.round(((dist['Negativo'] || 0) / total) * 100);
        this.neutralPercentage = Math.round(((dist['Neutro'] || 0) / total) * 100);
    }
}
