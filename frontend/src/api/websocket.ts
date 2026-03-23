import { useEffect, useRef, useState, useCallback } from "react";

export interface PipelineMessage {
  type: string;
  message: string;
  data?: Record<string, unknown>;
  timestamp: string;
}

export interface UsePipelineWebSocket {
  messages: PipelineMessage[];
  send: (data: Record<string, unknown>) => void;
  isConnected: boolean;
  lastResult: PipelineMessage | null;
  clearMessages: () => void;
}

export function usePipelineWebSocket(): UsePipelineWebSocket {
  const [messages, setMessages] = useState<PipelineMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [lastResult, setLastResult] = useState<PipelineMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Tracks whether the hook is still mounted — prevents reconnect storms from
  // React Strict Mode's double-invoke or component unmounting mid-flight.
  const mountedRef = useRef(false);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const ws = new WebSocket(`${protocol}//${host}/ws/pipeline`);

    ws.onopen = () => {
      if (!mountedRef.current) {
        ws.close();
        return;
      }
      setIsConnected(true);
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
    };

    ws.onmessage = (event) => {
      try {
        const msg: PipelineMessage = JSON.parse(event.data);
        setMessages((prev) => [...prev, msg]);
        if (msg.type === "result" || msg.type === "complete" || msg.type === "error") {
          setLastResult(msg);
        }
      } catch {
        setMessages((prev) => [
          ...prev,
          { type: "raw", message: event.data, timestamp: new Date().toISOString() },
        ]);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      wsRef.current = null;
      // Only reconnect if the hook is still mounted
      if (mountedRef.current) {
        reconnectTimer.current = setTimeout(connect, 3000);
      }
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setLastResult(null);
  }, []);

  return { messages, send, isConnected, lastResult, clearMessages };
}
