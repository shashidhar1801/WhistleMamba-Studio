"use strict";
// Magnitudes are display samples, not calibrated SPL or model probabilities.
// Keep source levels immutable; each tile is recolored at most once per revision.
class SpectrogramAppearance {
  constructor(redraw) {
    this.$=id=>document.getElementById(id);this.redraw=redraw;this.tables={};
    this.defaults={specView:"power",specColor:"magma",contrast:100,brightness:0,highlights:0};
    this.key="whistlemamba.spectrogram.v1";this.revision=0;
    const $=this.$;let saved={};
    try{saved=JSON.parse(localStorage.getItem(this.key)||"{}")||{};}catch{}
    for(const [id,def] of Object.entries(this.defaults)){
      const e=$(id),v=saved[id];
      if(typeof def==="number")e.value=Number.isFinite(v)?Math.max(+e.min,Math.min(+e.max,v)):def;
      else e.value=[...e.options].some(o=>o.value===v)?v:def;
      e.addEventListener(typeof def==="number"?"input":"change",()=>this.changed());
    }
    $("toneToggle").onclick=()=>{const p=$("tonePanel");p.hidden=!p.hidden;$("toneToggle").setAttribute("aria-expanded",String(!p.hidden));};
    $("resetAppearance").onclick=()=>{for(const [id,def] of Object.entries(this.defaults))$(id).value=def;this.changed();};
  }
  init(palettes){
    for(const [key,value] of Object.entries(palettes||{})){
      const lut=Uint8Array.from(atob(value),v=>v.charCodeAt(0));if(lut.length===768)this.tables[key]=lut;
    }
    if(!this.tables.magma)throw Error("The spectrogram palette is unavailable. Reload the updated local editor.");
    this.changed(false);
  }
  changed(save=true){
    const $=this.$,values={};for(const [id,def] of Object.entries(this.defaults))values[id]=typeof def==="number"?+$(id).value:$(id).value;
    this.values=values;this.revision++;
    const signed=x=>x>0?"+"+x:String(x);
    $("contrastValue").value=(values.contrast/100).toFixed(2)+"×";$("brightnessValue").value=signed(values.brightness);$("highlightsValue").value=signed(values.highlights);
    for(const id of ["contrast","brightness","highlights"])$(id).setAttribute("aria-valuetext",$(id+"Value").value);
    const tonal=values.specView==="tonal";
    $("viewDescription").textContent=tonal?"Tonal detail · peaks above nearby frequencies":"Standard power · fixed scale across the recording";
    $("scaleLow").textContent=tonal?"Low":"−110 dB";$("scaleHigh").textContent=tonal?"High":"−10 dB";
    if(save){try{localStorage.setItem(this.key,JSON.stringify(values));}catch{}}
    const color=this.tables[values.specColor];if(!color)return;
    const lut=this.lut=new Uint8Array(3*256);
    for(let i=0;i<256;i++){
      const v=i/255;
      // Upper-half-only adjustment; bounded slope keeps the scale monotonic.
      const shoulder=Math.max(0,2*v-1);
      const lifted=v+.48*(values.highlights/100)*shoulder;
      const tone=Math.min(1,Math.max(0,(lifted-.5)*values.contrast/100+.5+values.brightness/100));
      const src=3*Math.round(255*tone);
      lut[i*3]=color[src];lut[i*3+1]=color[src+1];lut[i*3+2]=color[src+2];
    }
    const key=$("colorKey"),k=key.getContext("2d");
    for(let x=0;x<key.width;x++){const i=Math.round(x*255/(key.width-1))*3;k.fillStyle="rgb("+lut[i]+","+lut[i+1]+","+lut[i+2]+")";k.fillRect(x,0,1,key.height);}
    if(!this.frame)this.frame=requestAnimationFrame(()=>{this.frame=0;this.redraw();});
  }
  decode(img){
    const c=document.createElement("canvas");c.width=img.width;c.height=img.height;
    const ctx=c.getContext("2d",{willReadFrequently:true});ctx.drawImage(img,0,0);
    const pixels=ctx.getImageData(0,0,c.width,c.height),raw=new Uint8Array(c.width*c.height);
    for(let i=0;i<raw.length;i++)raw[i]=pixels.data[4*i];
    return {c,ctx,pixels,raw,revision:-1};
  }
  tonal(d){
    if(d.detail)return d.detail;
    const w=d.c.width,h=d.c.height,raw=d.raw,out=new Uint8Array(raw.length);
    // Compare each time column with a centered nine-row frequency neighborhood.
    // No time normalization: the result does not depend on the visible viewport.
    for(let x=0;x<w;x++){let sum=0;
      for(let y=0;y<Math.min(h,4);y++)sum+=raw[y*w+x];
      for(let y=0;y<h;y++){
        if(y+4<h)sum+=raw[(y+4)*w+x];if(y>4)sum-=raw[(y-5)*w+x];
        const n=Math.min(h-1,y+4)-Math.max(0,y-4)+1,i=y*w+x;
        out[i]=Math.min(255,Math.round(.18*raw[i]+4*Math.max(0,raw[i]-sum/n)));
      }
    }
    return d.detail=out;
  }
  image(d){
    if(d.revision!==this.revision){
      const idx=this.values.specView==="tonal"?this.tonal(d):d.raw,p=d.pixels.data,lut=this.lut;
      for(let i=0,j=0;i<idx.length;i++,j+=4){const n=idx[i]*3;p[j]=lut[n];p[j+1]=lut[n+1];p[j+2]=lut[n+2];}
      d.ctx.putImageData(d.pixels,0,0);d.revision=this.revision;
    }
    return d.c;
  }
}
