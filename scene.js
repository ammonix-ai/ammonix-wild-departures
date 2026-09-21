const canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d');
const C={ink:'#123238',muted:'#66817e',paper:'#f5f2e9',mint:'#bdecc5',green:'#238a67',blue:'#b9dce7',orange:'#f5bb85',line:'#d7dfd6',dark:'#0d292f',red:'#c75446'};
let DATA=null,images={},timeline=[],duration=1,current=0,playing=false,start=0;
function rr(x,y,w,h,r=12,fill=C.paper,stroke=null){if(w<=0||h<=0)return;r=Math.min(r,w/2,h/2);ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fillStyle=fill;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke()}}
function text(s,x,y,size=20,color=C.ink,weight=400,align='left'){ctx.fillStyle=color;ctx.font=`${Math.round(weight/100)*100} ${size}px "Segoe UI",Arial,sans-serif`;ctx.textAlign=align;ctx.fillText(String(s),x,y)}
function answerLines(value,width,size){
  ctx.font=`400 ${size}px "Segoe UI",Arial,sans-serif`;
  const lines=[];
  for(const paragraph of String(value).replace(/\r\n?/g,'\n').split('\n')){
    if(!paragraph){lines.push('');continue}
    let line='';
    for(const word of paragraph.split(/\s+/)){
      const next=line?`${line} ${word}`:word;
      if(ctx.measureText(next).width<=width){line=next;continue}
      if(line){lines.push(line);line=''}
      if(ctx.measureText(word).width<=width){line=word;continue}
      for(const character of word){if(line&&ctx.measureText(line+character).width>width){lines.push(line);line=''}line+=character}
    }
    lines.push(line);
  }
  return lines;
}
function fullAnswer(value,x,y,width,height){
  // Fit every word of the original answer, including explicit line breaks.
  let size=18,lines=answerLines(value,width,size),leading=size*1.22;
  while(lines.length*leading>height&&size>6){size-=.5;leading=size*1.22;lines=answerLines(value,width,size)}
  lines.forEach((line,i)=>text(line,x,y+size+i*leading,size,C.ink,400));
}
const mammalClasses=new Set(['dog','cat','horse','cow','sheep','goat','pig','elephant','lion','tiger','bear','rabbit','giraffe','zebra','monkey','kangaroo']);
const birdClasses=new Set(['eagle','owl','chicken','duck','parrot','penguin','flamingo','ostrich']);
const reptileClasses=new Set(['crocodile','alligator','snake','turtle','lizard']);
const arthropodClasses=new Set(['bee','butterfly','ant','beetle','grasshopper','dragonfly','ladybug']);
function animalCategory(label){const species=String(label||'').toLowerCase();return mammalClasses.has(species)?'MAMMAL':birdClasses.has(species)?'BIRD':reptileClasses.has(species)?'REPTILE':arthropodClasses.has(species)?'ARTHROPOD':''}
const traditionalTime=row=>row?.traditional_ms??row?.qwen_ms??0;
const traditionalAnswer=row=>row?.traditional_answer??row?.qwen_answer??'';
function line(x,y,x2,y2,color=C.line,width=1){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x2,y2);ctx.stroke()}
function dot(x,y,r,color){ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill()}
function photo(im,x,y,w,h,r=14){if(!im)return;ctx.save();ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.clip();ctx.fillStyle='#e4e8df';ctx.fillRect(x,y,w,h);const scale=Math.min(w/im.width,h/im.height);ctx.drawImage(im,x+(w-im.width*scale)/2,y+(h-im.height*scale)/2,im.width*scale,im.height*scale);ctx.restore()}
const ease=t=>1-Math.pow(1-Math.max(0,Math.min(1,t)),3), ms=v=>v==null?'—':`${Math.round(v)} ms`;
function plane(x,y,scale=1){ctx.save();ctx.translate(x,y);ctx.scale(scale,scale);ctx.fillStyle=C.ink;ctx.beginPath();ctx.moveTo(-45,3);ctx.lineTo(-7,-2);ctx.lineTo(-20,-28);ctx.lineTo(-12,-28);ctx.lineTo(12,-3);ctx.lineTo(37,0);ctx.quadraticCurveTo(48,3,37,6);ctx.lineTo(10,8);ctx.lineTo(-13,29);ctx.lineTo(-21,29);ctx.lineTo(-7,8);ctx.lineTo(-34,9);ctx.lineTo(-45,3);ctx.fill();ctx.restore()}
function icon(type,x,y,s,color=C.ink){ctx.save();ctx.translate(x,y);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=s*.07;ctx.lineCap='round';if(type===0){dot(-s*.2,-s*.17,s*.12,color);dot(s*.05,-s*.25,s*.12,color);dot(s*.29,-s*.12,s*.11,color);ctx.beginPath();ctx.ellipse(.03*s,.16*s,.26*s,.20*s,-.2,0,7);ctx.fill()}else if(type===1){ctx.beginPath();ctx.moveTo(-.4*s,.06*s);ctx.quadraticCurveTo(-.13*s,-.36*s,0,0);ctx.quadraticCurveTo(.18*s,-.36*s,.4*s,.06*s);ctx.stroke()}else{ctx.beginPath();ctx.ellipse(0,0,.15*s,.29*s,0,0,7);ctx.fill();for(let i=-1;i<=1;i++){line(-.12*s,i*.17*s,-.35*s,(i*.17+.1)*s,color,s*.045);line(.12*s,i*.17*s,.35*s,(i*.17+.1)*s,color,s*.045)}}ctx.restore()}
function airport(t,row,phase){
  // Glass curtain wall, distant runway and deliberately playful boarding gates.
  rr(38,192,1524,437,25,'#e6eeea');
  const grad=ctx.createLinearGradient(0,190,0,420);grad.addColorStop(0,'#d1e8e8');grad.addColorStop(1,'#f4f2e9');ctx.fillStyle=grad;ctx.fillRect(40,218,1520,194);
  for(let i=0;i<10;i++){rr(60+i*173,343-(i%3)*18,80,71+(i%3)*18,2,'#cfddda');line(45+i*172,218,45+i*172,411,'#b7d0cd',3)}
  plane(730+Math.sin(t*.07)*140,267,.6);
  for(let i=0;i<4;i++){ctx.globalAlpha=.5;rr(400+i*230,244+(i%2)*36,85,9,6,'#fff');ctx.globalAlpha=1}
  ctx.fillStyle='#e4e7de';ctx.fillRect(40,409,1520,218);
  for(let x=-300;x<2100;x+=140)line(800+(x-800)*.35,409,x,628,'#d1d9cf');
  for(let y of [436,474,523,584])line(40,y,1560,y,'#d1d9cf');
  const groups=['MAMMALS','BIRDS & REPTILES','ARTHROPODS'],colors=[C.mint,C.blue,C.orange];
  const selected=row?groups.indexOf(row.escalated?row.final_gate:row.gate):-1;
  for(let i=0;i<3;i++){
    const x=910+i*209,active=phase==='routed'&&selected===i;
    rr(x,319,180,48,8,C.dark);text(`0${i+1}`,x+14,350,22,colors[i],700);text(groups[i],x+103,348,i===1?12:15,'#fff',650,'center');
    rr(x+9,376,162,164,9,active?colors[i]:'#d1ded7',active?C.green:'#bccdc3');
    rr(x+18,386,144,140,5,'#eef3ed');
    const open=active?ease(Math.min(1,(phase==='routed'?1:0))):0;
    rr(x+19,388,70*(1-open),136,3,'#b8cbc1');rr(x+92+70*open,388,70*(1-open),136,3,'#c6d6cd');
    if(!active)icon(i,x+90,451,48,'#6b877c');else{text('→',x+90,470,61,C.green,400,'center')}
    rr(x+11,526,158,15,4,active?C.green:'#839d90');
    dot(x+156,339,4,active?'#9cf6bd':'#6d8c82');
    text(active?'CLEARED TO BOARD':'AWAITING PASSENGER',x+90,563,10,active?C.green:C.muted,700,'center');
  }
  // Camera arch and lane markings.
  rr(563,343,174,207,16,'#d1dcd3');rr(579,363,142,184,8,'#eaf0e7');
  rr(613,335,74,30,8,C.dark);dot(650,350,8,'#8dbaae');dot(650,350,3,'#123238');
  text('VISION SCAN',650,325,12,C.muted,700,'center');
  if(phase==='scanning'){let sy=380+((t*130)%130);line(590,sy,710,sy,C.green,3);ctx.fillStyle='#64cc9930';ctx.fillRect(585,sy-23,130,23)}
  line(416,576,825,576,'#9aafa2',3);for(let x=450;x<805;x+=80)text('›',x,583,29,'#869e91',500);
  text('ALL CREATURES WELCOME',90,589,13,C.muted,700);
}
function renderFrame(time){
  current=Math.max(0,Math.min(duration,time));ctx.clearRect(0,0,1600,900);ctx.fillStyle=C.paper;ctx.fillRect(0,0,1600,900);
  let item=timeline.find(i=>current>=i.start&&current<i.end)||timeline.at(-1);let row=item?.row,local=current-(item?.start||0),intro=item?.kind==='intro',outro=item?.kind==='outro';
  const timing=row?.ammonix_ms??0,gen=traditionalTime(row);
  const phase=!row?'waiting':local<.22?'scanning':local<.22+timing/1000?'scanning':row.escalated&&local<.22+(timing+(row.fallback_ms||0))/1000?'escalating':'routed';
  ctx.fillStyle=C.dark;ctx.fillRect(0,0,1600,98);dot(52,48,16,C.mint);text('a',52,56,27,C.dark,750,'center');text('ammonix',79,56,27,'#fff',650);line(236,28,236,69,'#426064');text('WILD DEPARTURES',264,45,16,C.mint,700);text('Animal security · Terminal 01',264,69,14,'#b8cbc7');
  rr(1218,25,342,46,8,'#244147');dot(1238,48,4,C.mint);text(DATA?.partial?'MEASURED PREVIEW · RUN IN PROGRESS':DATA?.measured?'MEASURED INFERENCE REPLAY':'DESIGN PREVIEW · NO RESULTS',1253,53,12,'#d7e8db',650);
  text(intro?'A little expertise. A lot of departures.':outro?(DATA?.partial?'Benchmark still running.':'Fast decisions. Thoughtful exceptions.'):row?.ood?'One passenger outside the training set.':'Seventy arrivals. Three gates.',40,146,32,C.ink,650);
  text('Ammonix  vs  Traditional LLM   ·   Same image. Same model. Same GPU.',42,175,16,C.muted,450);
  airport(current,row,phase);
  // Passenger boarding pass.
  rr(66,222,345,331,16,'#ffffff');rr(66,222,345,42,16,C.dark);ctx.fillStyle=C.dark;ctx.fillRect(66,244,345,20);
  text(row?.ood?'SPECIAL ASSISTANCE':'PASSENGER IDENTIFICATION',86,249,12,C.mint,700);text(row?String(row.index+1).padStart(2,'0'):'—',390,250,15,'#fff',650,'right');
  if(row){photo(images[row.image],82,277,313,194,8);
    text('GOLD LABEL · REFERENCE',87,488,10,C.green,750);
    text(row.species.toUpperCase(),86,518,27,C.ink,700);
    text(animalCategory(row.species),87,540,13,C.muted,600);
  }else{icon(0,239,366,100,'#bdd5c4');text(intro?'35 animal classes':'TERMINAL READY',238,480,28,C.ink,650,'center');text(intro?'70 held-out arrivals · 3 gates':'Waiting for a measured run',238,513,15,C.muted,500,'center')}
  // A moving photo pass is the animal travelling through the checkpoint.
  if(row){let move=phase==='routed'?ease((local-.22-timing/1000-(row.escalated?(row.fallback_ms||0)/1000:0))/.7):0;const groups=['MAMMALS','BIRDS & REPTILES','ARTHROPODS'];const gate=groups.indexOf(row.escalated?row.final_gate:row.gate);if(gate<0)move=0;let px=650+(Math.max(0,gate)*209+1000-650)*move;let py=482-25*move;
    ctx.save();ctx.globalAlpha=1-.85*move;rr(px-46,py-53,92,96,10,'#fff');photo(images[row.image],px-41,py-48,82,72,6);text('✈',px,py+38,13,C.ink,500,'center');ctx.restore();
    if(phase==='escalating'){rr(557,566,204,34,8,C.orange);text('SYSTEM TWO REQUESTED',659,589,12,C.ink,700,'center')}
  }
  // Independent results stay separate from the photo's reference label.
  const reveal=phase!=='scanning'&&row,traditionalReady=row&&local>=.22+gen/1000;
  const elapsed=Math.max(0,local-.22)*1000;
  rr(38,650,450,188,16,'#e9eee4');text('AMMONIX',61,681,15,C.green,750);
  text(reveal?ms(timing):'—',465,681,25,C.ink,650,'right');
  rr(61,694,404,5,3,'#dbe4d9');rr(61,694,row?404*Math.min(1,elapsed/Math.max(timing,1)):0,5,3,C.green);
  text(reveal&&row.escalated?'DECISION TO ESCALATE':'STRUCTURED DECISION',61,719,10,C.muted,700);
  text(reveal?(row.escalated?'UNKNOWN':row.prediction.toUpperCase()):row?'SCANNING…':'READY TO SCAN',61,750,27,C.ink,700);
  const groupIndex=row?['MAMMALS','BIRDS & REPTILES','ARTHROPODS'].indexOf(row.gate):-1;
  const reviewed=reveal&&row.escalated&&phase==='routed'&&row.final_label;
  const reviewLabel=reviewed?`SYSTEM TWO → ${row.final_label.toUpperCase()}${animalCategory(row.final_label)?' · '+animalCategory(row.final_label):''}`:'SYSTEM TWO REVIEW';
  text(reveal?(row.escalated?reviewLabel:`${animalCategory(row.prediction)}  ·  GATE 0${groupIndex+1}`):'Image → decision',62,773,13,reviewed?C.green:row?.escalated?C.red:C.muted,600);
  text('Class score',62,798,12,C.muted);text(reveal?`${(row.confidence*100).toFixed(1)}%`:'—',465,798,18,C.ink,650,'right');
  if(reveal&&row.escalated){const status=reviewed?`FULL RESOLUTION · ${ms(timing+(row.fallback_ms||0))} TOTAL`:'Uncertainty threshold triggered';text(status,62,821,11,reviewed?C.green:C.red,650)}
  else text(reveal?'Species → group → gate':'Known expertise. Fast decisions.',62,821,11,C.muted,500);
  rr(508,650,723,188,16,'#e9eee4');text('TRADITIONAL LLM',531,681,15,C.ink,750);
  text(traditionalReady?ms(gen):'—',1208,681,25,C.ink,650,'right');
  rr(531,694,677,5,3,'#dbe4d9');rr(531,694,row?677*Math.min(1,elapsed/Math.max(gen,1)):0,5,3,'#88aebd');
  text('FULL ANSWER · UNPARSED',531,719,10,C.muted,700);
  fullAnswer(traditionalReady?traditionalAnswer(row):row?'Generating the animal species and group…':'The complete model response will appear here.',531,728,677,96);
  rr(1251,650,311,188,16,C.dark);text('DEPARTURES',1274,681,12,C.mint,700);
  const available=DATA?.rows?.filter(r=>!r.ood).length||0;const processed=row?row.index+(phase==='routed'?1:0):(outro?available:0);const n=Math.min(processed,70,available);const completed=DATA?.rows?.slice(0,n)||[];
  text(`${String(n).padStart(2,'0')} / 70`,1274,733,39,'#fff',600);text('HELD-OUT IMAGES',1275,757,11,'#a9c6bd',700);
  text(`Ammonix correct: ${completed.filter(r=>r.correct).length}`,1274,786,16,C.mint,600);text(`${completed.filter(r=>r.escalated).length} escalated`,1274,814,14,'#c0d3ce',500);
  text(`${n} / 70 IMAGES   ·   AMMONIX ${completed.filter(r=>r.correct).length} CORRECT   ·   AVG ${n?ms(completed.reduce((s,r)=>s+r.ammonix_ms,0)/n):'—'}`,730,54,12,'#c5dcd2',600);
  text('AMMONIX  /  35 ANIMAL CLASSES  /  SYSTEM ONE + SYSTEM TWO',40,868,11,C.muted,650);
  text('Sequential GPU measurements · replayed together · source credits supplied',1560,868,11,C.muted,400,'right');
  ctx.fillStyle=C.green;ctx.fillRect(0,895,1600*current/duration,5);
  if(intro){text('01',1471,142,33,C.green,700,'right')}
  if(outro&&DATA?.report){ctx.save();ctx.fillStyle='#f5f2e9ed';ctx.fillRect(38,192,1524,437);const r=DATA.report; text('AMMONIX · THE MEASURED RESULT',800,266,13,C.green,750,'center');text(`${(r.top1_accuracy*100).toFixed(1)}% top-1 accuracy`,800,333,49,C.ink,650,'center');text(`Median Ammonix  ${ms(r.median_ammonix_ms)}     |     Traditional LLM  ${ms(r.median_traditional_ms??r.median_qwen_ms)}`,800,391,23,C.ink,500,'center');text(`Macro-F1 ${r.macro_f1.toFixed(3)}   ·   Macro AUROC ${r.macro_ovr_auroc.toFixed(3)}   ·   70 held-out images`,800,441,18,C.muted,500,'center');text('Known expertise → System One. Uncertainty → System Two.',800,513,25,C.ink,600,'center');text('Small demonstration set; results describe this run and this hardware.',800,561,14,C.muted,400,'center');ctx.restore()}
  document.querySelector('#seek').value=1000*current/duration;document.querySelector('#status').textContent=`${Math.floor(current/60)}:${String(Math.floor(current%60)).padStart(2,'0')} / ${Math.floor(duration/60)}:${String(Math.floor(duration%60)).padStart(2,'0')}`;
}
function makeTimeline(){timeline=[{kind:'intro',start:0,end:4}];let t=4;for(const [i,r]of DATA.rows.entries()){r.index=i;const actual=Math.max(r.ammonix_ms+(r.escalated?r.fallback_ms||0:0),traditionalTime(r))/1000;const words=traditionalAnswer(r).trim().split(/\s+/).filter(Boolean).length;const reading=Math.max(1.25,words/6);const len=Math.max(i<3?4:1.5,actual+reading,r.ood?6:0);timeline.push({row:r,start:t,end:t+len});t+=len}timeline.push({kind:'outro',start:t,end:t+7});duration=t+7;window.duration=duration}
async function init(){try{const response=await fetch('results.json');if(response.ok){DATA=await response.json()}else{const partial=await fetch('benchmark-progress.json');if(!partial.ok)throw Error('No measured results yet');DATA={measured:true,partial:true,rows:await partial.json()};document.querySelector('#mode').textContent='Measured preview · benchmark still running'}makeTimeline();await Promise.all(DATA.rows.map(r=>new Promise(resolve=>{let im=new Image();im.onload=resolve;im.onerror=resolve;im.src=r.image;images[r.image]=im})));document.querySelector('#play').disabled=false;document.querySelector('#restart').disabled=false;document.querySelector('#play').textContent='Play demo';window.ready=true;renderFrame(0)}catch(e){duration=1;renderFrame(0);document.querySelector('#play').textContent='Awaiting benchmark';document.querySelector('#mode').textContent='No performance claims until measured results are available';window.ready=true}}
function tick(now){if(playing){renderFrame((now-start)/1000);if(current>=duration){playing=false;document.querySelector('#play').textContent='Replay'}else requestAnimationFrame(tick)}}
document.querySelector('#play').onclick=()=>{playing=!playing;if(playing){if(current>=duration)current=0;start=performance.now()-current*1000;requestAnimationFrame(tick)}document.querySelector('#play').textContent=playing?'Pause':'Play demo'};
document.querySelector('#restart').onclick=()=>{current=0;start=performance.now();renderFrame(0)};
document.querySelector('#seek').oninput=e=>{playing=false;document.querySelector('#play').textContent='Play demo';renderFrame(e.target.value*duration/1000)};
window.renderFrame=renderFrame;init();
if(typeof document.createElement==='function'){
  fetch('/api/status').then(r=>r.ok?r.json():null).then(status=>{
    if(!status?.live)return;
    const live=document.createElement('button');live.textContent='Measure this image again';document.querySelector('nav').appendChild(live);
    live.onclick=async()=>{
      playing=false;const item=timeline.find(i=>current>=i.start&&current<i.end&&i.row)||timeline.find(i=>i.row);if(!item)return;
      live.disabled=true;live.textContent='Running both approaches…';document.querySelector('#mode').textContent='Fresh GPU inference in progress';
      try{const res=await fetch('/api/classify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:item.row.id})});if(!res.ok)throw Error(await res.text());const fresh=await res.json();Object.assign(item.row,fresh);makeTimeline();const updated=timeline.find(t=>t.row?.id===fresh.id);current=updated.start;start=performance.now()-current*1000;playing=true;requestAnimationFrame(tick);document.querySelector('#play').textContent='Pause';document.querySelector('#mode').textContent='Fresh measurement replay · single trial';}
      catch(e){document.querySelector('#mode').textContent='Live inference failed: '+e.message}
      finally{live.disabled=false;live.textContent='Measure this image again'}
    };
  }).catch(()=>{});
}
