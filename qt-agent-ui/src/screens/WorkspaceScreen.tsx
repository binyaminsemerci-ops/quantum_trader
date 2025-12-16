import { useState, useEffect } from "react";
import { WidgetShell, ScreenGrid, span } from "../components/Widget";
import { Search, Plus, CheckCircle, Circle, Settings, Database, FileText } from "lucide-react";
import { useMetrics, useModelInfo } from "../hooks/useData";

const DEFAULT_TASKS = [
  { id: 1, text: "Review daily trading performance", completed: false },
  { id: 2, text: "Adjust risk parameters", completed: false },
  { id: 3, text: "Analyze AI signals accuracy", completed: true },
  { id: 4, text: "Update trading strategy", completed: false },
];

export default function WorkspaceScreen() {
  const { data: metrics } = useMetrics();
  const { data: modelInfo } = useModelInfo();
  const [searchTerm, setSearchTerm] = useState("");
  const [newTaskText, setNewTaskText] = useState("");
  const [showAddTask, setShowAddTask] = useState(false);
  const [tasks, setTasks] = useState<Array<{id: number, text: string, completed: boolean}>>(() => {
    const saved = localStorage.getItem('quantum-tasks');
    return saved ? JSON.parse(saved) : DEFAULT_TASKS;
  });

  useEffect(() => {
    localStorage.setItem('quantum-tasks', JSON.stringify(tasks));
  }, [tasks]);

  const toggleTask = (id: number) => {
    setTasks(tasks.map(task => 
      task.id === id ? { ...task, completed: !task.completed } : task
    ));
  };

  const addTask = () => {
    if (newTaskText.trim()) {
      setTasks([...tasks, { id: Date.now(), text: newTaskText.trim(), completed: false }]);
      setNewTaskText("");
      setShowAddTask(false);
    }
  };

  const deleteTask = (id: number) => {
    setTasks(tasks.filter(task => task.id !== id));
  };

  const exportTradeHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/positions');
      const data = await response.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `trades_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export failed:', err);
      alert('Failed to export trade history');
    }
  };

  const viewSignals = () => {
    // Navigate to signals view - for now just alert
    alert('Opening signals view...');
  };

  const viewModelDetails = () => {
    // Navigate to model details - for now just alert
    alert('Model Details:\n' + JSON.stringify(modelInfo, null, 2));
  };

  return (
    <div className="mx-auto max-w-[1280px] px-4 pb-24 pt-4">
      <ScreenGrid>
        {/* Search Widget */}
        <WidgetShell title="Quick Search" className={`${span.half} h-[180px]`}>
          <div className="flex flex-col h-full">
            <div className="relative mb-3">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search symbols, signals, trades..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-lg bg-black/5 dark:bg-white/5 border border-slate-200 dark:border-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="text-xs text-slate-500 space-y-1">
              <div>• Press / to focus search</div>
              <div>• Type symbol to filter signals</div>
              <div>• Use filters: type:buy, conf:&gt;80</div>
            </div>
          </div>
        </WidgetShell>

        {/* Notes Widget */}
        <WidgetShell title="Trading Notes" className={`${span.half} h-[180px]`}>
          <div className="flex flex-col h-full">
            <textarea
              placeholder="Add trading notes, observations, strategy adjustments..."
              className="flex-1 w-full p-2 rounded-lg bg-black/5 dark:bg-white/5 border border-slate-200 dark:border-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm resize-none"
            />
            <button className="mt-2 self-end px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors">
              Save Note
            </button>
          </div>
        </WidgetShell>

        {/* Tasks Widget */}
        <WidgetShell title="Tasks" className={`${span.third} h-[240px]`}>
          <div className="flex flex-col h-full">
            <div className="flex-1 overflow-y-auto space-y-2 pr-1 min-h-0">
              {tasks.map((task) => (
                <div 
                  key={task.id}
                  className="group flex items-start gap-2 p-2 rounded hover:bg-black/5 dark:hover:bg-white/5"
                >
                  <div 
                    className="flex-1 flex items-start gap-2 cursor-pointer"
                    onClick={() => toggleTask(task.id)}
                  >
                    {task.completed ? (
                      <CheckCircle className="w-4 h-4 text-emerald-500 flex-shrink-0 mt-0.5" />
                    ) : (
                      <Circle className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5" />
                    )}
                    <span className={`text-sm flex-1 ${task.completed ? "line-through text-slate-400" : ""}`}>
                      {task.text}
                    </span>
                  </div>
                  <button
                    onClick={() => deleteTask(task.id)}
                    className="opacity-0 group-hover:opacity-100 text-rose-500 hover:text-rose-600 text-xs flex-shrink-0"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
            {showAddTask ? (
              <div className="mt-2 flex gap-2">
                <input
                  type="text"
                  value={newTaskText}
                  onChange={(e) => setNewTaskText(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && addTask()}
                  placeholder="New task..."
                  autoFocus
                  className="flex-1 px-2 py-1 text-xs rounded bg-black/5 dark:bg-white/5 border border-slate-200 dark:border-slate-700 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
                <button 
                  onClick={addTask}
                  className="px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  Add
                </button>
                <button 
                  onClick={() => { setShowAddTask(false); setNewTaskText(""); }}
                  className="px-2 py-1 text-xs bg-slate-200 dark:bg-slate-700 rounded hover:bg-slate-300 dark:hover:bg-slate-600"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button 
                onClick={() => setShowAddTask(true)}
                className="mt-2 flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700"
              >
                <Plus className="w-3 h-3" />
                Add Task
              </button>
            )}
          </div>
        </WidgetShell>

        {/* Datasets Widget */}
        <WidgetShell title="Datasets" className={`${span.third} h-[240px]`}>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-2 rounded-lg bg-black/5 dark:bg-white/5">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-blue-500" />
                <div>
                  <div className="text-sm font-medium">Trade History</div>
                  <div className="text-xs text-slate-500">{metrics?.total_trades || 0} records</div>
                </div>
              </div>
              <button 
                onClick={exportTradeHistory}
                className="text-xs text-blue-600 hover:underline"
              >
                Export
              </button>
            </div>

            <div className="flex items-center justify-between p-2 rounded-lg bg-black/5 dark:bg-white/5">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-emerald-500" />
                <div>
                  <div className="text-sm font-medium">AI Signals</div>
                  <div className="text-xs text-slate-500">Latest 1000</div>
                </div>
              </div>
              <button 
                onClick={viewSignals}
                className="text-xs text-blue-600 hover:underline"
              >
                View
              </button>
            </div>

            <div className="flex items-center justify-between p-2 rounded-lg bg-black/5 dark:bg-white/5">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-purple-500" />
                <div>
                  <div className="text-sm font-medium">Model Training</div>
                  <div className="text-xs text-slate-500">
                    {modelInfo?.training_date ? new Date(modelInfo.training_date).toLocaleDateString("no-NO") : "Not trained"}
                  </div>
                </div>
              </div>
              <button 
                onClick={viewModelDetails}
                className="text-xs text-blue-600 hover:underline"
              >
                Details
              </button>
            </div>
          </div>
        </WidgetShell>

        {/* Settings Widget */}
        <WidgetShell title="Quick Settings" className={`${span.third} h-[240px]`}>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Settings className="w-4 h-4 text-slate-400" />
                <span className="text-sm">Autonomous Mode</span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" className="sr-only peer" defaultChecked={metrics?.autonomous_mode} />
                <div className="w-9 h-5 bg-slate-200 peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer dark:bg-slate-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-emerald-600"></div>
              </label>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Settings className="w-4 h-4 text-slate-400" />
                <span className="text-sm">Dry Run Mode</span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" className="sr-only peer" defaultChecked />
                <div className="w-9 h-5 bg-slate-200 peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer dark:bg-slate-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-sky-600"></div>
              </label>
            </div>

            <div className="pt-2 border-t border-slate-200 dark:border-slate-700">
              <div className="text-xs text-slate-500 space-y-1">
                <div>Total Trades: {metrics?.total_trades || 0}</div>
                <div>Win Rate: {(metrics?.win_rate || 0).toFixed(1)}%</div>
                <div>Active Positions: {metrics?.positions_count || 0}</div>
              </div>
            </div>

            <button className="w-full mt-2 py-2 text-sm bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded transition-colors">
              Advanced Settings →
            </button>
          </div>
        </WidgetShell>
      </ScreenGrid>
    </div>
  );
}
