import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
    SearchQuery,
    SocialComment,
    Statistics,
    Storytelling,
    ApiResponse
} from '../models/models';

@Injectable({
    providedIn: 'root'
})
export class ApiService {
    private baseUrl = 'http://localhost:8000/api';

    constructor(private http: HttpClient) { }

    /**
     * Iniciar proceso de scraping y análisis
     */
    startScraping(searchQuery: SearchQuery): Observable<ApiResponse<any>> {
        return this.http.post<ApiResponse<any>>(`${this.baseUrl}/scrape`, searchQuery);
    }

    /**
     * Obtener resultados de análisis
     */
    getResults(taskId: string, platform?: string): Observable<ApiResponse<SocialComment[]>> {
        let params = new HttpParams();
        if (platform) {
            params = params.set('platform', platform);
        }

        return this.http.get<ApiResponse<SocialComment[]>>(
            `${this.baseUrl}/results/${taskId}`,
            { params }
        );
    }

    /**
     * Obtener estadísticas agregadas
     */
    getStatistics(taskId: string): Observable<Statistics> {
        return this.http.get<Statistics>(`${this.baseUrl}/stats/${taskId}`);
    }

    /**
     * Obtener storytelling (narrativa de resultados)
     */
    getStorytelling(taskId: string): Observable<Storytelling> {
        return this.http.get<Storytelling>(`${this.baseUrl}/storytelling/${taskId}`);
    }

    /**
     * Health check del backend
     */
    healthCheck(): Observable<any> {
        return this.http.get(`${this.baseUrl.replace('/api', '')}/`);
    }
}
