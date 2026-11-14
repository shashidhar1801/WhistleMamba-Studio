"use strict";
const el=id=>document.getElementById(id),records=new Map();
let token="",model=false,current=null,timer=null,uploading=false;
function status(msg,error=false){el("status").textContent=msg;el("status").className=error?"bad":"";}
function guard(fn){return async(...args)=>{try{await fn(...args);}catch(e){status(e.message,true);}};}
async function api(path,data={},blob=false){
  const res=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-Studio-Token":token},body:JSON.stringify(data)});
  if(!res.ok)throw Error((await res.json()).error||"Request failed.");return blob?res.blob():res.json();
}
const tl=new RecordingTimeline(api,status);
function list(){
  el("count").textContent=records.size;el("list").replaceChildren();el("emptylist").hidden=!!records.size;
  for(const [id,r] of records){const btn=document.createElement("button");btn.className="file"+(id===current?" active":"");btn.textContent=r.name;btn.title=r.name;btn.onclick=guard(()=>select(id));el("list").append(btn);}
}
function controls(r){
  const selected=r.tracks.find(c=>c.id===el("track").value);tl.track=selected?.id;
  el("notes").value=selected?.notes||"";el("pointIndex").max=selected?.n||1;el("pointIndex").value=1;
  el("other").replaceChildren(...r.tracks.filter(c=>c.id!==selected?.id).map(c=>new Option(c.source+" · "+c.start.toFixed(2)+" s · "+c.id.slice(0,6),c.id)));
  for(const btn of document.querySelectorAll("[data-op]")){
    const op=btn.dataset.op;btn.disabled=r.phase!=="done"||(op==="undo"?!r.undo:op==="redo"?!r.redo:
      op==="merge"?!el("other").value:op==="new"?false:!selected);
  }
}
function detail(r){
  el("name").textContent=r.name;el("meta").textContent=r.seconds.toFixed(2)+" s  ·  "+r.rate/1000+" kHz  ·  "+r.channels+" channel(s)";
  const busy=r.phase==="running",done=r.phase==="done";
  el("analysisLocation").textContent=model?"Analysis runs locally on the GPU.":"Recording preview";
  const sampleBand=el("recordingBand");
  sampleBand.textContent=r.rate===96000?"96 kHz recording · analysis band 4–48 kHz. Accuracy at this source rate has not been evaluated.":"";
  sampleBand.hidden=!sampleBand.textContent;
  el("eligibility").textContent=!r.compatible?"Preview only. Detection requires ≥4 s at 96 kHz or ≥192 kHz.":!model?"Preview only: no local model configured.":r.rate===96000?"":r.mode_notice||"";
  el("run").disabled=!model||!r.compatible||busy||done;el("run").textContent=done?"Analysis complete":r.phase==="cancelled"?"Resume analysis":"Run analysis";
  el("cancel").disabled=!busy;el("save").disabled=!done;el("remove").disabled=busy;el("tracker").disabled=el("batch").disabled=busy||done;
  if(r.options){el("tracker").value=r.options.tracker;el("batch").value=r.options.batch;}
  el("progress").textContent=r.progress||"";
  el("trackInfo").textContent=done?r.tracks.length+" contours. Selecting a contour takes you to its position.":"Automatic contours appear after analysis.";
  const old=el("track").value;el("track").replaceChildren(...r.tracks.map((c,i)=>new Option((i+1)+". "+c.source+" · "+c.start.toFixed(2)+"–"+c.end.toFixed(2)+" s · "+c.state,c.id)));
  if(r.tracks.some(c=>c.id===old))el("track").value=old;
  controls(r);el("editPane").hidden=!done;el("track").hidden=!r.tracks.length;
}
function remember(r){records.set(r.id,r);if(r.id===current){detail(r);tl.update(r);}list();}
function poll(){
  clearTimeout(timer);if(![...records.values()].some(r=>r.phase==="running"))return;
  timer=setTimeout(async()=>{try{for(const r of [...records.values()])if(r.phase==="running"){
    const now=await api("/r/"+r.id+"/state");if(records.has(r.id))remember(now);
  }}catch(e){status(e.message,true);}finally{poll();}},1000);
}
async function select(id){
  current=id;list();el("welcome").hidden=true;el("detail").hidden=false;el("tracker").value="learned";el("batch").value=1;
  const r=records.get(id);detail(r);tl.open(r);status("");
  const fresh=await api("/r/"+id+"/state");if(current!==id)return;remember(fresh);poll();
}
async function upload(chosen){
  if(uploading)return;const files=Array.from(chosen).filter(f=>/\.wav$/i.test(f.name));if(!files.length)return status("Choose at least one WAV.",true);
  uploading=true;let first=null,added=0;const issues=[];
  try{for(const f of files){status("Opening "+f.name+"…");if(f.size>512*1024**2){issues.push(f.name+": exceeds 512 MiB");continue;}
    const res=await fetch("/recordings",{method:"POST",headers:{"Content-Type":"application/octet-stream","X-Studio-Token":token,"X-Wav-Name":encodeURIComponent(f.name)},body:f});
    const row=await res.json();if(!res.ok){issues.push(f.name+": "+row.error);continue;}
    records.set(row.id,row);first??=row.id;added++;list();}
    if(first)await select(first);status(added+" recording(s) opened. "+issues.join("; "),!!issues.length);
  }finally{uploading=false;el("files").value=el("folder").value="";}
}
el("pick").onclick=el("browse").onclick=()=>el("files").click();el("folderBtn").onclick=()=>el("folder").click();
el("files").onchange=el("folder").onchange=guard(e=>upload(e.target.files));
for(const n of ["dragover","drop"])document.addEventListener(n,e=>{if(!e.dataTransfer.types.includes("Files"))return;e.preventDefault();document.body.classList.toggle("drag",n==="dragover");if(n==="drop")guard(upload)(e.dataTransfer.files);});
document.addEventListener("dragleave",e=>{if(!e.relatedTarget)document.body.classList.remove("drag");});
el("run").onclick=guard(async()=>{const id=current;el("run").disabled=true;try{remember(await api("/r/"+id+"/run",{tracker:el("tracker").value,batch:+el("batch").value}));poll();}finally{if(current===id)detail(records.get(id));}});
el("cancel").onclick=guard(async()=>remember(await api("/r/"+current+"/cancel")));
el("track").onchange=()=>{const r=records.get(current),c=r.tracks.find(c=>c.id===el("track").value);controls(r);if(c){tl.pan(c.start-tl.span()/4);if(tl.a.readyState>=1)tl.a.currentTime=c.start;}};
for(const btn of document.querySelectorAll("[data-op]"))btn.onclick=guard(async()=>{
  const op=btn.dataset.op;if(op==="delete"&&!confirm("Delete this contour from the review? You can undo this edit."))return;
  const id=current,r=await api("/r/"+id+"/edit",{op,id:el("track").value,state:btn.dataset.value,other:el("other").value,
    notes:el("notes").value,index:+el("pointIndex").value-1,time:+el("pointTime").value,frequency:+el("pointFreq").value*1000});
  remember(r);if(current===id)status("Edit applied to the session copy.");
});
el("save").onclick=guard(async()=>{const r=records.get(current);status("Preparing review bundle…");const b=await api("/r/"+r.id+"/export",{},true),url=URL.createObjectURL(b),a=document.createElement("a");a.href=url;a.download=r.name.replace(/\.wav$/i,"")+"-review.zip";a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);status("Review bundle downloaded.");});
el("remove").onclick=guard(async()=>{if(!confirm("Remove this session copy and its edits? Your original WAV will not change."))return;
  const id=current;await api("/r/"+id+"/remove");records.delete(id);if(current!==id){list();return;}current=null;tl.open(null);
  if(records.size)await select(records.keys().next().value);else{el("detail").hidden=true;el("welcome").hidden=false;list();}status("Session copy removed.");});
guard(async()=>{const s=await(await fetch("/session")).json();tl.appearance.init(s.displayPalettes);token=s.token;model=s.model;s.recordings.forEach(r=>records.set(r.id,r));list();if(records.size)await select(records.keys().next().value);poll();})();
