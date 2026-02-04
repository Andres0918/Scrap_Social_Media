import { Injectable } from '@angular/core';
import { Observable, Observer, Subject } from 'rxjs';
import { WebSocketMessage } from '../models/models';

@Injectable({
    providedIn: 'root'
})
export class WebsocketService {
    private socket: WebSocket | null = null;
    private messagesSubject = new Subject<WebSocketMessage>();
    private wsUrl = 'ws://localhost:8000/ws';
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectInterval = 3000;

    constructor() { }

    /**
     * Conectar al WebSocket
     */
    connect(): Observable<WebSocketMessage> {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            return this.messagesSubject.asObservable();
        }

        this.socket = new WebSocket(this.wsUrl) as WebSocket;

        this.socket.onopen = () => {
            console.log('✅ WebSocket conectado');
            this.reconnectAttempts = 0;
        };

        this.socket.onmessage = (event) => {
            try {
                const message: WebSocketMessage = JSON.parse(event.data);
                this.messagesSubject.next(message);
            } catch (error) {
                console.error('Error parseando mensaje WebSocket:', error);
            }
        };

        this.socket.onerror = (error) => {
            console.error('❌ Error en WebSocket:', error);
        };

        this.socket.onclose = () => {
            console.log('🔌 WebSocket desconectado');
            this.attemptReconnect();
        };

        return this.messagesSubject.asObservable();
    }

    /**
     * Intentar reconexión automática
     */
    private attemptReconnect(): void {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`🔄 Reintentando conexión (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

            setTimeout(() => {
                this.connect();
            }, this.reconnectInterval);
        } else {
            console.error('❌ Máximo de reintentos alcanzado');
        }
    }

    /**
     * Enviar mensaje al WebSocket
     */
    sendMessage(message: any): void {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(message));
        } else {
            console.warn('⚠️ WebSocket no está conectado');
        }
    }

    /**
     * Desconectar WebSocket
     */
    disconnect(): void {
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
    }

    /**
     * Obtener estado de conexión
     */
    getConnectionState(): number {
        return this.socket?.readyState ?? WebSocket.CLOSED;
    }

    /**
     * Verificar si está conectado
     */
    isConnected(): boolean {
        return this.socket?.readyState === WebSocket.OPEN;
    }
}
