import { useState } from 'react';

interface ChatMessage {
  timestamp: string;
  message: string;
  system?: string;
  user?: string;
}

export default function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      timestamp: new Date().toISOString(),
      message: "AI Assistant ready. Ask me about trading strategies or market analysis.",
      system: "AI"
    }
  ]);
  const [input, setInput] = useState('');

  const sendMessage = async (message: string) => {
    if (!message.trim()) return;

    // Add user message
    const userMessage: ChatMessage = {
      timestamp: new Date().toISOString(),
      message: message.trim(),
      user: "You"
    };
    setMessages(prev => [...prev, userMessage]);

    // Simulate AI response
    setTimeout(() => {
      const responses = [
        "Based on current market conditions, I recommend monitoring BTC resistance levels.",
        "The enhanced data feeds show bullish sentiment across multiple indicators.",
        "Consider the Fear & Greed index at 52 (Neutral) for position sizing.",
        "AI continuous learning is active - model accuracy improving.",
        "Current volatility suggests good opportunities for swing trading."
      ];
      
      const aiResponse: ChatMessage = {
        timestamp: new Date().toISOString(),
        message: responses[Math.floor(Math.random() * responses.length)],
        system: "AI"
      };
      setMessages(prev => [...prev, aiResponse]);
    }, 1000);
  };

  const handleSend = () => {
    sendMessage(input);
    setInput('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 h-96 flex flex-col">
      <h3 className="text-lg font-semibold text-white mb-4">💬 AI Trading Assistant</h3>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-2 mb-4">
        {messages.map((msg, index) => (
          <div key={index} className={`p-2 rounded ${
            msg.system ? 'bg-blue-900/30 border-l-2 border-blue-400' : 'bg-gray-700'
          }`}>
            <div className="flex justify-between items-start mb-1">
              <span className={`text-xs font-medium ${
                msg.system ? 'text-blue-400' : 'text-green-400'
              }`}>
                {msg.system || msg.user}
              </span>
              <span className="text-xs text-gray-400">
                {new Date(msg.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <div className="text-sm text-white">
              {msg.message}
            </div>
          </div>
        ))}
      </div>

      {/* Input */}
      <div className="flex space-x-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about trading strategies..."
          className="flex-1 bg-gray-700 text-white px-3 py-2 rounded border border-gray-600 focus:border-blue-400 focus:outline-none"
        />
        <button
          onClick={handleSend}
          disabled={!input.trim()}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded"
        >
          Send
        </button>
      </div>
    </div>
  );
}