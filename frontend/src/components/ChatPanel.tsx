import { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

interface ChatMessage {
  timestamp?: string;
  message: string;
  system?: string;
}

export default function ChatPanel() {
  const { data, sendMessage, connectionStatus } = useWebSocket<ChatMessage | ChatMessage[]>({
    url: 'ws://127.0.0.1:8000/ws/chat',
    enabled: true,
    debounceMs: 0
  });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');

  useEffect(() => {
    if (!data) return;
    if (Array.isArray(data)) {
      setMessages(prev => [...prev, ...data]);
    } else {
      setMessages(prev => [...prev, data]);
    }
  }, [data]);

  const send = () => {
    if (!input.trim()) return;
    sendMessage({ text: input.trim() });
    setInput('');
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-lg font-semibold text-fuchsia-400">Chat</h2>
        <span className="text-xs text-gray-400">{connectionStatus}</span>
      </div>
      <div className="flex-1 overflow-y-auto space-y-1 text-xs pr-2">
        {messages.slice(-200).map((m,i) => (
          <div key={i} className="border-b border-gray-700 pb-1">
            <span className="text-gray-500 mr-2">{m.timestamp ? new Date(m.timestamp).toLocaleTimeString() : ''}</span>
            {m.system ? (
              <span className="text-yellow-400">{m.message}</span>
            ) : (
              <span>{m.message || (m as any).text}</span>
            )}
          </div>
        ))}
      </div>
      <div className="mt-2 flex space-x-2">
        <input
          className="flex-1 bg-gray-700 rounded px-2 py-1 text-sm focus:outline-none"
          value={input}
          onChange={e=>setInput(e.target.value)}
          onKeyDown={e=> { if (e.key==='Enter') send(); }}
          placeholder="Write a message..."
        />
        <button onClick={send} className="px-3 py-1 bg-fuchsia-600 hover:bg-fuchsia-700 rounded text-sm">Send</button>
      </div>
    </div>
  );
}
