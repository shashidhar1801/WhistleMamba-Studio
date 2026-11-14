"use strict";
// Render from absolute recording time, not detector windows. Adjacent cached
// strips occupy disjoint intervals. Playback advances only the shared camera.
class RecordingTimeline {
  constructor(api, report) {
    this.api=api; this.report=report; this.$=id=>document.getElementById(id);
    const $=this.$; this.cv=$("scope");this.a=$("audio");this.cache=new Map();this.gen=0;
    this.start=0;this.shapes=[];this.tracksOn=true;this.rej=false;
    this.appearance=new SpectrogramAppearance(()=>this.paint());
    new ResizeObserver(()=>this.paint()).observe(this.cv);
    this.a.onplay=()=>this.tick();this.a.ontimeupdate=this.a.onseeked=()=>this.follow();
    this.a.onerror=()=>{if(this.row)this.report("Chrome could not play this WAV encoding. Spectrogram scrolling and analysis remain available.",true);};
    $("windowPos").oninput=e=>this.pan(+e.target.value);
    $("next").onclick=()=>this.pan(this.start+this.span());$("back").onclick=()=>this.pan(this.start-this.span());
    $("follow").onchange=()=>this.follow();
    $("span").onchange=()=>this.place(this.start);
    $("lines").onchange=e=>{this.tracksOn=e.target.checked;this.paint();};
    $("rejected").onchange=e=>{this.rej=e.target.checked;this.paint();};
    $("setBand").onclick=()=>{const lo=+$("low").value*1000,hi=+$("high").value*1000;
      if(!(this.floor<=lo&&lo<hi&&hi<=this.ceil))return this.report("Choose a frequency range inside "+this.floor/1000+"–"+this.ceil/1000+" kHz.",true);
      this.low=lo;this.high=hi;this.paint();};
    this.cv.onpointerdown=e=>{if(!this.row||e.button!==0)return;this.grab={x:e.clientX,start:this.start,moved:false};this.cv.setPointerCapture(e.pointerId);};
    this.cv.onpointermove=e=>{if(!this.grab)return;const dx=e.clientX-this.grab.x;if(Math.abs(dx)>3)this.grab.moved=true;if(this.grab.moved)this.pan(this.grab.start-dx*this.span()/this.geom().pw);};
    this.cv.onpointercancel=()=>this.grab=null;
    this.cv.onpointerup=e=>{if(!this.grab)return;const click=!this.grab.moved;this.grab=null;if(!click)return;
      const r=this.cv.getBoundingClientRect(),g=this.geom(),x=e.clientX-r.left,y=e.clientY-r.top;
      if(x<g.l||x>g.l+g.pw)return;
      const t=Math.min(this.row.seconds,Math.max(0,this.start+(x-g.l)/g.pw*this.span()));
      $("pointTime").value=t.toFixed(4);if(y>=g.s&&y<=g.b)$("pointFreq").value=((this.high-(y-g.s)/(g.b-g.s)*(this.high-this.low))/1000).toFixed(3);
      if(this.a.readyState>=1)this.a.currentTime=t;this.paint();};
    this.cv.addEventListener("wheel",e=>{if(!this.row||(!e.deltaX&&!e.shiftKey))return;e.preventDefault();this.pan(this.start+(e.deltaX||e.deltaY)*this.span()/this.geom().pw);},{passive:false});
  }
  geom(){const w=this.cv.clientWidth,h=this.cv.clientHeight;return{w,h,l:52,pw:Math.max(1,w-64),s:63,b:h-30};}
  span(){return Math.min(this.row.seconds,+this.$("span").value);}
  open(row){
    this.a.pause();this.gen++;this.cache.clear();this.shapes=[];this.shapeKey=null;this.track=null;this.row=row;this.start=0;
    if(!row){this.a.removeAttribute("src");this.a.load();return;}
    this.ceil=Math.min(96000,row.rate/2);this.floor=this.ceil>4100?4000:0;this.low=this.floor;this.high=this.ceil;
    this.$("low").value=this.low/1000;this.$("high").value=this.high/1000;this.$("span").value="15";this.$("follow").checked=true;
    this.a.src="/media/"+row.id;this.place(0);
  }
  update(row){if(row.id!==this.row?.id)return;this.row=row;this.shapeKey=null;this.place(this.start);}
  pan(t){this.$("follow").checked=false;this.place(t);}
  follow(){if(!this.row)return;const t=this.a.currentTime;if(this.$("follow").checked &&
      (t<this.start||t>=this.start+this.span()*0.65))this.place(Math.max(0,t-this.span()*0.65));else this.paint();}
  tick(){cancelAnimationFrame(this.frame);const loop=()=>{if(!this.row||this.a.paused)return;this.follow();this.frame=requestAnimationFrame(loop);};loop();}
  place(t){
    if(!this.row)return;
    const span=this.span(),dur=this.row.seconds;this.start=Math.max(0,Math.min(t,dur-span));const bar=this.$("windowPos");
    bar.max=Math.max(0,dur-span);bar.value=this.start;bar.disabled=dur<=span;
    const a=Math.floor(this.start/5),b=Math.min(Math.ceil(dur/5)-1,Math.ceil((this.start+span)/5)-1);
    for(let i=a;i<=b;i++)this.fetchTile(i);
    if(b+1<Math.ceil(dur/5))this.fetchTile(b+1);
    for(const i of this.cache.keys())if(i<a-2||i>b+3)this.cache.delete(i);
    const key=[this.row.id,a,b].join(":");
    if(this.row.phase==="done"&&this.shapeKey!==key){
      this.shapeKey=key;const gen=this.gen;
      this.api("/r/"+this.row.id+"/shapes",{start:a*5,end:Math.min(dur,(b+1)*5)}).then(shapes=>{
        if(this.gen===gen&&this.shapeKey===key){this.shapes=shapes;this.paint();}
      }).catch(e=>{if(this.gen===gen)this.report(e.message,true);});
    }
    this.paint();
  }
  fetchTile(i){if(this.cache.has(i))return;const slot={};this.cache.set(i,slot);const gen=this.gen;
    this.api("/r/"+this.row.id+"/tile",{index:i}).then(async d=>{if(this.gen!==gen||this.cache.get(i)!==slot)return;
      const img=new Image();img.src="data:image/png;base64,"+d.png;await img.decode();
      if(this.gen===gen&&this.cache.get(i)===slot){delete d.png;Object.assign(slot,d,{raster:this.appearance.decode(img)});this.paint();}
    }).catch(e=>{if(this.gen===gen){slot.error=true;this.report("Could not load part of the recording: "+e.message,true);this.paint();}});
  }
  paint(){
    if(!this.row||this.cv.clientWidth<1)return;
    const {w,h,l,pw,s,b}=this.geom(),cv=this.cv,dpr=Math.min(window.devicePixelRatio||1,2);
    if(cv.width!==Math.round(w*dpr)||cv.height!==Math.round(h*dpr)){cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);}
    const ctx=cv.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle="#f8faf9";ctx.fillRect(0,0,w,h);
    const span=this.span(),x=t=>l+(t-this.start)*pw/span,y=f=>b-(f-this.low)*(b-s)/(this.high-this.low);
    ctx.fillStyle="#0f232d";ctx.fillRect(l,6,pw,49);ctx.fillRect(l,s,pw,b-s);
    ctx.save();ctx.beginPath();ctx.rect(l,0,pw,b);ctx.clip();let missing=false;
    for(const [i,t] of this.cache){if(i*5>=this.start+span||(i+1)*5<=this.start)continue;
      if(!t.raster){missing=true;continue;}
      const img=this.appearance.image(t.raster);
      const sy=(t.high-this.high)/(t.high-t.low)*img.height,sh=(this.high-this.low)/(t.high-t.low)*img.height;
      ctx.drawImage(img,0,sy,img.width,sh,x(t.start),s,x(t.end)-x(t.start),b-s);
      ctx.strokeStyle="#6ad8bc";ctx.lineWidth=1;ctx.beginPath();
      t.wave.forEach(([lo,hi],j)=>{const px=x(t.start+(j+.5)*t.step);ctx.moveTo(px,30-22*Math.min(1,hi));ctx.lineTo(px,31-22*Math.max(-1,lo));});ctx.stroke();
    }
    ctx.save();ctx.beginPath();ctx.rect(l,s,pw,b-s);ctx.clip();
    if(this.tracksOn)for(const c of this.shapes){if(c.state==="rejected"&&!this.rej)continue;
      ctx.strokeStyle=c.state==="rejected"?"#999":c.id===this.track?"#fff":c.source==="heuristic"?"#ffbc56":"#53f6ee";ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=c.id===this.track?3:1.8;
      ctx.beginPath();c.points.forEach(([t,f],j)=>{if(j)ctx.lineTo(x(t),y(f));else ctx.moveTo(x(t),y(f));});ctx.stroke();
      if(c.points.length===1){const [t,f]=c.points[0];ctx.fillRect(x(t)-2,y(f)-2,4,4);}
    }ctx.restore();
    const cursor=x(this.a.currentTime||0);ctx.strokeStyle="#fff";ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(cursor,6);ctx.lineTo(cursor,b);ctx.stroke();ctx.setLineDash([]);ctx.restore();
    ctx.fillStyle="#314d52";ctx.font="12px system-ui";ctx.textAlign="right";ctx.fillText("Wave",l-6,32);
    for(let j=0;j<=4;j++){let f=this.low+(this.high-this.low)*j/4;ctx.fillText((f/1000).toFixed(1),l-6,Math.min(b,y(f)+5));}
    ctx.fillText("kHz",l-6,s-5);for(let j=0;j<=5;j++){ctx.textAlign=j===5?"right":j===0?"left":"center";const t=this.start+span*j/5;ctx.fillText(t.toFixed(1)+" s",x(t),b+18);}
    const label=this.start.toFixed(1)+"–"+(this.start+span).toFixed(1)+" s of "+this.row.seconds.toFixed(1)+" s"+(missing?" · loading display…":"");if(this.$("range").textContent!==label)this.$("range").textContent=label;
  }
}
