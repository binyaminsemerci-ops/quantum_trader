import { useEffect, useState } from 'react';

export default function DebugOverlay() {
  const [rootClasses, setRootClasses] = useState('');
  const [bodyClasses, setBodyClasses] = useState('');
  useEffect(() => {
    const update = () => {
      setRootClasses(document.documentElement.className);
      setBodyClasses(document.body.className);
    };
    update();
    const id = setInterval(update, 1500);
    return () => clearInterval(id);
  }, []);
  return (
    <div style={{position:'fixed',bottom:8,right:8,zIndex:9999,fontSize:11,background:'rgba(0,0,0,0.55)',color:'#fff',padding:'6px 8px',borderRadius:6,fontFamily:'monospace',maxWidth:260, lineHeight:1.3}}>
      <div style={{opacity:0.8}}>Debug</div>
      <div><strong>html:</strong> {rootClasses || '(none)'}</div>
      <div><strong>body:</strong> {bodyClasses || '(none)'}</div>
    </div>
  );
}
