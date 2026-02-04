// ================================
// SEARCH QUERY MODEL
// ================================
export interface SearchQuery {
    query: string;
    platforms: string[];
    max_comments?: number;
}

// ================================
// COMMENT MODEL (Renombrado para evitar conflicto con DOM Comment)
// ================================
export interface SocialComment {
    texto_original: string;
    origen: string;
    modelo: string;
    sentimiento: 'Positivo' | 'Negativo' | 'Neutro' | string;
    explicacion: string;
    tiempo_ejecucion: number;
    timestamp?: Date;
}

// ================================
// STATISTICS MODEL
// ================================
export interface Statistics {
    task_id: string;
    query: string;
    stats: {
        total_comments: number;
        sentiment_distribution: { [key: string]: number };
        platform_breakdown: { [key: string]: number };
        avg_processing_time: number;
    };
    timestamp: Date;
}

// ================================
// WEBSOCKET MESSAGE MODEL
// ================================
export interface WebSocketMessage {
    task_id: string;
    platform?: string;
    status: 'scraping' | 'nlp_processing' | 'completed' | 'error' | 'aggregating' | 'finished';
    message?: string;
    comments_found?: number;
    progress?: string;
    total_analyzed?: number;
    error?: string;
    stats?: any;
}

// ================================
// TASK MODEL
// ================================
export interface Task {
    task_id: string;
    query: string;
    platforms: string[];
    status: string;
    timestamp: Date;
}

// ================================
// STORYTELLING MODEL
// ================================
export interface Storytelling {
    overall_sentiment: string;
    platform_comparison: {
        [platform: string]: {
            dominant_sentiment: string;
            total_comments: number;
            distribution: { [sentiment: string]: number };
            avg_length: number;
        };
    };
    key_insights: {
        type: string;
        message: string;
    }[];
    sentiment_trends: any;
}

// ================================
// PLATFORM MODEL
// ================================
export interface Platform {
    id: string;
    name: string;
    icon: string;
    color: string;
    enabled: boolean;
}

// ================================
// CHART DATA MODEL
// ================================
export interface ChartData {
    labels: string[];
    datasets: {
        label?: string;
        data: number[];
        backgroundColor?: string | string[];
        borderColor?: string | string[];
        borderWidth?: number;
    }[];
}

// ================================
// API RESPONSE MODEL
// ================================
export interface ApiResponse<T> {
    task_id?: string;
    message?: string;
    total?: number;
    platform?: string;
    data?: T;
    comments?: T; // Added to match backend response
    error?: string;
}
