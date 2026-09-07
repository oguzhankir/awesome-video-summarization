// Parse/render every mathematical expression and parse every Mermaid diagram.
// KaTeX is a compatibility screen; GitHub uses its own MathJax rendering.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {JSDOM} from 'jsdom';
import katex from 'katex';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dom = new JSDOM('<!doctype html><html><body></body></html>');
globalThis.window = dom.window;
globalThis.document = dom.window.document;
const {default: mermaid} = await import('mermaid');
mermaid.initialize({startOnLoad:false, securityLevel:'strict'});
function walk(dir) {
  return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e => {
    if (['node_modules','.git','.venv','__pycache__'].includes(e.name)) return [];
    const p = path.join(dir,e.name);
    return e.isDirectory() ? walk(p) : p.endsWith('.md') ? [p] : [];
  });
}
let mathematics=0, diagrams=0;
const failures=[];
for (const file of walk(root)) {
  const source=fs.readFileSync(file,'utf8');
  const lines=source.split('\n');
  let fence=null, body=[], math=null;
  const render=(text,displayMode,line)=>{
    try { katex.renderToString(text,{throwOnError:true,strict:'error',displayMode}); mathematics++; }
    catch(error) { failures.push(`${path.relative(root,file)}:${line}: ${error.message}`); }
  };
  for (let i=0;i<lines.length;i++) {
    const line=lines[i];
    const marker=line.match(/^\s{0,3}(`{3,}|~{3,})(.*)$/);
    if (marker) {
      if (!fence) {fence={char:marker[1][0],size:marker[1].length,language:marker[2].trim(),line:i+1};body=[];}
      else if (marker[1][0]===fence.char && marker[1].length>=fence.size && !marker[2].trim()) {
        if (fence.language==='mermaid') {
          try { await mermaid.parse(body.join('\n'));diagrams++; }
          catch(error) { failures.push(`${path.relative(root,file)}:${fence.line}: ${error.message}`); }
        }
        if (fence.language==='math') render(body.join('\n'),true,fence.line);
        fence=null;
      } else body.push(line);
      continue;
    }
    if(fence){body.push(line);continue;}
    if(line.trim()==='$$') {
      if(math===null) math={line:i+1,text:[]};
      else {render(math.text.join('\n'),true,math.line);math=null;}
      continue;
    }
    if(math!==null){math.text.push(line);continue;}
    // GitHub math spans use $`formula`$ as well as ordinary $formula$.
    // Parse them before dropping ordinary inline code.
    const withoutMathCode=line.replace(/(?<!\\)\$(`+)([\s\S]*?)\1\$/g, (_match,_ticks,formula)=>{render(formula,false,i+1);return '';});
    const noCode=withoutMathCode.replace(/`+[^`]*`+/g,'');
    if(/(?<!\\)\\[\[\]]/.test(noCode)) failures.push(`${path.relative(root,file)}:${i+1}: use $$ display math instead of raw backslash brackets`);
    for(const match of noCode.matchAll(/(?<!\\)\$([^$]+?)(?<!\\)\$/g))render(match[1],false,i+1);
  }
  if(fence) failures.push(`${path.relative(root,file)}:${fence.line}: unclosed code fence`);
  if(math) failures.push(`${path.relative(root,file)}:${math.line}: unclosed display math`);
}
if(failures.length){console.error(failures.join('\n'));process.exit(1);}
console.log(`PASS: ${mathematics} math expressions rendered; ${diagrams} Mermaid diagrams parsed`);
console.log('LIMIT: this does not certify GitHub browser hydration or layout; use the Preview checklist.');
