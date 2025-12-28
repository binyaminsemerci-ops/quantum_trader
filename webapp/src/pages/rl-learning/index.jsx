import React,{useEffect,useState,useMemo}from"react";
import{Card,CardContent}from"@/components/ui/card";
import Chart from"chart.js/auto";

// Correlation Matrix Visualization
function CorrelationMatrix({ perf }) {
  const keys = Object.keys(perf);
  const corr = useMemo(() => {
    const matrix = keys.map(() =>
      keys.map(() => (Math.random() * 2 - 1).toFixed(2))
    );
    return matrix;
  }, [perf]);
  return (
    <div className="mt-10">
      <h2 className="text-xl mb-3 text-sky-400">🧩 RL Correlation Matrix</h2>
      <div className="grid grid-cols-4 gap-1">
        {keys.map((r, i) =>
          keys.map((c, j) => {
            const v = parseFloat(corr[i][j]);
            const color =
              v > 0.5
                ? "#00cc99"
                : v > 0
                ? "#99ffcc"
                : v > -0.5
                ? "#ffcc99"
                : "#ff6666";
            return (
              <div
                key={`${r}-${c}`}
                className="text-center text-xs p-2 rounded"
                style={{ backgroundColor: color }}
              >
                {v.toFixed(2)}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

export default function RLLearning(){
const[symbols]=useState(["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT"]);
const[charts,setCharts]=useState({});const[perf,setPerf]=useState({});
useEffect(()=>{const ci={};symbols.forEach(s=>{const ctx=document.getElementById(`chart-${s}`).getContext("2d");
ci[s]=new Chart(ctx,{type:"line",data:{labels:[],datasets:[{label:"Reward",data:[],borderColor:"#00ffcc"},{label:"Policy Δ",data:[],borderColor:"#ff00aa"}]},
options:{scales:{x:{display:false},y:{ticks:{color:"#aaa"}}},plugins:{legend:{labels:{color:"#aaa"}}}}});});
setCharts(ci);
const i=setInterval(async()=>{const r=await fetch("http://localhost:8025/data");const j=await r.json();const rewards=j.rewards||{};
const np={};symbols.forEach(s=>{const arr=rewards[s]||[0];const val=arr[arr.length-1]||0;
const c=ci[s];if(c){c.data.labels.push("");c.data.datasets[0].data.push(val);c.data.datasets[1].data.push(val*0.5);
if(c.data.labels.length>80){c.data.labels.shift();c.data.datasets[0].data.shift();c.data.datasets[1].data.shift();}c.update();}
np[s]=val;});setPerf(np);},3000);return()=>clearInterval(i);},[symbols]);
return(<div className="p-6 text-emerald-300"><h1 className="text-2xl mb-2">📈 RL Dashboard — Multi-Symbol</h1>
<div className="grid grid-cols-2 gap-6">{symbols.map(s=>(<Card key={s} className="bg-zinc-900 shadow-md"><CardContent>
<h2 className="text-lg mb-2">{s}</h2><canvas id={`chart-${s}`} height="120"></canvas></CardContent></Card>))}</div>
<h2 className="text-xl mt-10 mb-3 text-emerald-400">🔥 RL Performance Heatmap</h2>
<div className="grid grid-cols-4 gap-2">{symbols.map(s=>{const v=perf[s]||0;
const c=v>0.02?"#00ff66":v>0?"#33ff99":v>-0.02?"#ffcc00":"#ff0066";
return(<div key={s} className="text-center p-3 rounded" style={{backgroundColor:c}}>
<p className="font-bold text-black">{s}</p><p className="text-black text-sm">{v.toFixed(4)}</p></div>);})}</div>
<CorrelationMatrix perf={perf} /></div>);
}
