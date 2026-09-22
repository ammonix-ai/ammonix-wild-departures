// Wild Departures: the animal airport of the film, driven by the live ternary model (or replaying the measured run).
const canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d');
// Corporate palette sampled from the supplied Ammonix brand references.
const C={ink:'#211d4b',muted:'#666b85',paper:'#f5f7fc',indigo:'#18106d',cobalt:'#4a5ac9',rose:'#b24a6d',lavender:'#8c66c0',blueTint:'#e9edfd',roseTint:'#fbf0f4',lavenderTint:'#f1ebfb',line:'#dce0f0',dark:'#18106d',paleBlue:'#cdd4ff',paleRose:'#f4c8d8'};
const GROUPS=['MAMMALS','BIRDS & REPTILES','ARTHROPODS'];
const LEAD=.22,HOLD=3,SLOTS=7;
let STATUS=null,PASSENGERS=null,REPLAY=null,images={},brandLogo=null,session=null,lastTick=null;
const $=id=>document.querySelector('#'+id);
const brandReady=new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>{brandLogo=im;resolve()};im.onerror=()=>reject(Error('The Ammonix logo could not be loaded.'));im.src='brand/ammonix-logo.png'});
function ammonite(x,y,width,opacity=1){if(!brandLogo)return;ctx.save();ctx.globalAlpha=opacity;ctx.drawImage(brandLogo,196,308,1227,935,x,y,width,width*935/1227);ctx.restore()}
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
function fullAnswer(value,x,y,width,height,color=C.ink){
  let size=18,lines=answerLines(value,width,size),leading=size*1.22;
  while(lines.length*leading>height&&size>6){size-=.5;leading=size*1.22;lines=answerLines(value,width,size)}
  lines.forEach((line,i)=>text(line,x,y+size+i*leading,size,color,400));
}
const mammalClasses=new Set(['dog','cat','horse','cow','sheep','goat','pig','elephant','lion','tiger','bear','rabbit','giraffe','zebra','monkey','kangaroo']);
const birdClasses=new Set(['eagle','owl','chicken','duck','parrot','penguin','flamingo','ostrich']);
const reptileClasses=new Set(['crocodile','alligator','snake','turtle','lizard']);
const arthropodClasses=new Set(['bee','butterfly','ant','beetle','grasshopper','dragonfly','ladybug']);
function animalCategory(label){const species=String(label||'').toLowerCase();return mammalClasses.has(species)?'MAMMAL':birdClasses.has(species)?'BIRD':reptileClasses.has(species)?'REPTILE':arthropodClasses.has(species)?'ARTHROPOD':''}
function line(x,y,x2,y2,color=C.line,width=1){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x2,y2);ctx.stroke()}
function dot(x,y,r,color){ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill()}
function photo(im,x,y,w,h,r=14){if(!im||!im.complete||!im.naturalWidth)return;ctx.save();ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.clip();ctx.fillStyle='#f0f1f8';ctx.fillRect(x,y,w,h);const scale=Math.min(w/im.naturalWidth,h/im.naturalHeight);ctx.drawImage(im,x+(w-im.naturalWidth*scale)/2,y+(h-im.naturalHeight*scale)/2,im.naturalWidth*scale,im.naturalHeight*scale);ctx.restore()}
const ease=t=>1-Math.pow(1-Math.max(0,Math.min(1,t)),3), ms=v=>v==null?'—':v<1000?`${Math.round(v)} ms`:`${(v/1000).toFixed(2)} s`;
const words=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty','twenty-one','twenty-two','twenty-three','twenty-four','twenty-five','twenty-six','twenty-seven','twenty-eight','twenty-nine','thirty'];
const numberWord=n=>n<words.length?words[n][0].toUpperCase()+words[n].slice(1):String(n);
const median=values=>{const v=values.filter(Number.isFinite).sort((a,b)=>a-b);return v.length?v[Math.floor((v.length-1)/2)]:null};
const gpuShort=()=>String(STATUS?.gpu||'this GPU').replace(/NVIDIA\s*/,'').replace(/\s*Workstation Edition/,'').split(',')[0].trim();
const rivalOf=row=>row?.rival||($('rival').hidden?'local':$('rival').value);
const rivalTitle=row=>rivalOf(row)==='gpt6'?['GPT6','MAX REASONING']:['TRADITIONAL LLM','SAME MODEL · FULL ANSWER'];
const rivalShort=row=>rivalOf(row)==='gpt6'?'GPT6':'LLM';

// ---- the session: a list of passengers passing through, with real (live) or measured (replay) times -----------------
function routedAt(item){return item.decisionAt==null?null:item.decisionAt+(item.row.escalated?Math.max(.6,Math.min(6,(item.row.fallback_ms||0)/1000)):0)}
function routedGate(row){return !row||row.prediction==null?null:row.escalated?row.final_gate||null:row.gate}
function currentItem(t){const items=session?.items||[];for(let i=items.length-1;i>=0;i--)if(t>=items[i].start)return items[i];return null}
function phaseOf(item,t){
  if(!item)return 'waiting';
  const local=t-item.start;
  if(local<LEAD||item.decisionAt==null)return 'scanning';
  if(item.row.escalated&&t<routedAt(item))return 'escalating';
  return 'routed';
}
function gateStats(t){
  const boarded=Object.fromEntries(GROUPS.map(g=>[g,0])),review=Object.fromEntries(GROUPS.map(g=>[g,0]));
  for(const item of session?.items||[]){
    const route=routedGate(item.row),at=routedAt(item);
    if(at==null)continue;
    if(!route){if(t>=at)review[item.row.gold_gate||GROUPS[0]]++;continue}
    if(t>=at+.7)boarded[route]++;
  }
  return{boarded,review};
}
function gateOccupants(t){
  const occupants=Object.fromEntries(GROUPS.map(g=>[g,[]]));
  for(const item of session?.items||[]){
    const route=routedGate(item.row),at=routedAt(item);
    if(!route||at==null)continue;
    if(t>=at+.7)occupants[route].push({label:item.row.escalated?item.row.final_label:item.row.prediction,boardedAt:at+.7});
  }
  return occupants;
}
function icon(type,x,y,s,color=C.ink){ctx.save();ctx.translate(x,y);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=s*.07;ctx.lineCap='round';if(type===0){dot(-s*.2,-s*.17,s*.12,color);dot(s*.05,-s*.25,s*.12,color);dot(s*.29,-s*.12,s*.11,color);ctx.beginPath();ctx.ellipse(.03*s,.16*s,.26*s,.20*s,-.2,0,7);ctx.fill()}else if(type===1){ctx.beginPath();ctx.moveTo(-.4*s,.06*s);ctx.quadraticCurveTo(-.13*s,-.36*s,0,0);ctx.quadraticCurveTo(.18*s,-.36*s,.4*s,.06*s);ctx.stroke()}else{ctx.beginPath();ctx.ellipse(0,0,.15*s,.29*s,0,0,7);ctx.fill();for(let i=-1;i<=1;i++){line(-.12*s,i*.17*s,-.35*s,(i*.17+.1)*s,color,s*.045);line(.12*s,i*.17*s,.35*s,(i*.17+.1)*s,color,s*.045)}}ctx.restore()}

function airport(t,row,phase){
  // Glass curtain wall, distant runway and deliberately playful boarding gates.
  rr(38,192,1524,437,25,'#edf0fa');
  const grad=ctx.createLinearGradient(0,190,0,420);grad.addColorStop(0,'#dde5fa');grad.addColorStop(1,'#f5f7fc');ctx.fillStyle=grad;ctx.fillRect(40,218,1520,194);
  for(let i=0;i<10;i++){rr(60+i*173,343-(i%3)*18,80,71+(i%3)*18,2,'#d3dced');line(45+i*172,218,45+i*172,411,'#bcc9e4',3)}
  for(let i=0;i<4;i++){ctx.globalAlpha=.5;rr(400+i*230,244+(i%2)*36,85,9,6,'#fff');ctx.globalAlpha=1}
  ctx.fillStyle='#eaedf5';ctx.fillRect(40,409,1520,218);
  for(let x=-300;x<2100;x+=140)line(800+(x-800)*.35,409,x,628,'#d7dceb');
  for(let y of [436,474,523,584])line(40,y,1560,y,'#d7dceb');
  const colors=[C.paleBlue,'#ded2f6',C.paleRose],accents=[C.cobalt,C.lavender,C.rose];
  const selected=GROUPS.indexOf(routedGate(row)),counts=gateStats(t),occupants=gateOccupants(t);
  const total=session?.queue?.length||PASSENGERS?.rows?.length||21,perGate=Math.max(SLOTS,Math.ceil(total/3));
  for(let i=0;i<3;i++){
    const x=910+i*209,active=phase==='routed'&&selected===i;
    rr(x,319,180,48,8,C.dark);text(`0${i+1}`,x+14,350,22,colors[i],700);text(GROUPS[i],x+103,348,i===1?12:15,'#fff',650,'center');
    rr(x+9,376,162,164,9,active?colors[i]:'#dfe3f2',active?C.cobalt:'#c3c9e1');
    rr(x+18,386,144,140,5,'#f8f9fe');
    const open=active?1:0;
    rr(x+19,388,70*(1-open),136,3,'#cbd1e5');rr(x+92+70*open,388,70*(1-open),136,3,'#dbe0ef');
    line(x+90,388,x+90,525,'#bec5de',1);
    const filled=occupants[GROUPS[i]],slotColor=colors[i],accent=accents[i];
    const visible=filled.slice(Math.max(0,filled.length-SLOTS));
    for(let slot=0;slot<SLOTS;slot++){
      const sy=508-slot*20,passenger=visible[slot];
      rr(x+22,sy,136,17,3,'#f9faff',passenger?accent:'#b9c2df');
      if(passenger){
        const fill=ease(Math.min(1,Math.max(0,(t-passenger.boardedAt)/.35)));
        rr(x+22,sy,136*fill,17,3,slotColor);rr(x+22,sy,4,17,2,accent);
        text(String(passenger.label).toUpperCase(),x+93,sy+12,11,C.indigo,700,'center');
      }else{text('—',x+90,sy+12,10,'#a6afcc',500,'center')}
    }
    rr(x+11,530,158,11,4,'#c7cde1');rr(x+11,530,158*Math.min(1,counts.boarded[GROUPS[i]]/perGate),11,4,accent);
    dot(x+156,339,4,active?'#e5e9ff':'#a4add6');
    text(`${counts.boarded[GROUPS[i]]} / ${perGate} BOARDED`,x+90,563,13,C.indigo,750,'center');
    text(active?'CLEARED TO BOARD':'BOARDING PLACES',x+90,582,9,C.muted,650,'center');
    if(counts.review[GROUPS[i]]){rr(x+17,591,146,24,6,C.roseTint,C.rose);text(`${counts.review[GROUPS[i]]} · REVIEW`,x+90,607,11,C.rose,750,'center')}
  }
  // Classifier distribution belongs at the scanner; the passenger lane stays below it.
  rr(442,319,426,216,16,'#ffffff',C.line);
  text('VISION SCAN',462,345,12,C.cobalt,750);
  text('35 CLASS PROBABILITIES · TOP 5',848,345,10,C.muted,650,'right');
  const probabilities=row?.probabilities,classes=STATUS?.classes;
  const chartReady=phase!=='scanning'&&Array.isArray(probabilities)&&probabilities.length===35&&Array.isArray(classes)&&classes.length===35;
  if(chartReady){
    const ranked=probabilities.map((value,index)=>({value:Number(value),label:classes[index]})).sort((a,b)=>b.value-a.value).slice(0,5);
    ranked.forEach((entry,i)=>{const y=368+i*29;const value=Math.max(0,Math.min(1,entry.value));
      text(String(entry.label).toUpperCase(),462,y+13,11,i?C.muted:C.ink,i?500:750);
      rr(568,y,224,17,5,'#e9ecf7');rr(568,y,224*value,17,5,i?['#8f9adb','#adb6e8','#c4caf0','#dce0f6'][i-1]:C.cobalt);
      text(`${(value*100).toFixed(1)}%`,848,y+13,11,i?C.muted:C.cobalt,650,'right');
    });
  }else{
    const message=phase==='scanning'?(session?.mode==='live'?'Reading the image on the GPU…':'Reading the image…'):row?'Awaiting classifier probabilities':'35 species. One compact decision layer.';
    text(message,655,430,17,C.muted,500,'center');
    if(phase==='scanning'){const sweep=(Math.sin(t*5)+1)/2;rr(474,452,362,4,2,'#e9ecf7');rr(474,452,362*sweep,4,2,C.cobalt)}
  }
  line(442,598,868,598,'#a8b1d7',3);for(let x=465;x<865;x+=80)text('›',x,605,29,'#8f9bc3',500);
  text('ALL CREATURES WELCOME',90,589,13,C.muted,700);
}

function renderFrame(t){
  ctx.clearRect(0,0,1600,900);ctx.fillStyle=C.paper;ctx.fillRect(0,0,1600,900);
  const item=currentItem(t),row=item?.row,local=item?t-item.start:0,phase=phaseOf(item,t);
  const live=session?.mode==='live',intro=!session||(session.items.length===0&&!session.done),outro=!!session?.done&&(!item||(item.end!=null&&t>=item.end));
  const decisionKnown=item?.decisionAt!=null,answerKnown=item?.answerAt!=null;
  const timing=decisionKnown?row.ammonix_ms:null;
  const ammonixElapsed=item?Math.max(0,(decisionKnown?item.decisionAt-item.start:local)-LEAD)*1000:0;
  const rivalStart=item?(live?(item.decisionAt??null):item.start+LEAD):null;
  const rivalElapsed=rivalStart==null?null:Math.max(0,(answerKnown?item.answerAt:t)-rivalStart)*1000;
  const gen=answerKnown?row.traditional_ms:null;
  // Header.
  ctx.fillStyle='#ffffff';ctx.fillRect(0,0,1600,98);
  ammonite(38,14,88);text('Ammonix',143,54,34,C.indigo,700);text('Agents For High-Stakes Applications',144,77,10,C.cobalt,500);
  line(342,25,342,73,C.line);text('WILD DEPARTURES',369,45,16,C.indigo,700);text('Animal security · Terminal 01',369,69,13,C.muted);
  line(0,98,1600,98,C.line);
  const pill=!STATUS?'CONNECTING TO THE TERMINAL':live?`LIVE · 1.75-BIT TERNARY MODEL · ${gpuShort().toUpperCase()}`:REPLAY?'REPLAY · MEASURED ON AN 8 GB LAPTOP GPU':'NO MODEL · NO MEASURED RUN';
  rr(1178,25,382,46,10,C.blueTint);dot(1198,48,4,live?C.rose:C.cobalt);text(pill,1213,53,11,C.indigo,650);
  const total=session?.queue?.length||STATUS?.passengers||21;
  text(intro?'Ultrafast stratification of our wild passengers.':outro?'Fast decisions. Thoughtful exceptions.':row?.upload?'A passenger of your own.':row?.ood?'One passenger outside the training set.':`${numberWord(total)} arrivals. Three gates.`,40,146,32,C.ink,650);
  const rivalName=rivalOf(row)==='gpt6'?'GPT6 (cloud, max reasoning)':'Traditional LLM';
  text(live?`Ammonix  vs  ${rivalName}   ·   Same 1.75-bit local model · Live on ${gpuShort()} · Single measurements`:`Ammonix  vs  Traditional LLM   ·   Same 1.75-bit local model · Replay of the measured laptop run · Medians of three trials`,42,175,15,C.muted,450);
  airport(t,row,phase);
  // Passenger boarding pass.
  rr(66,222,345,331,16,'#ffffff');rr(66,222,345,42,16,C.dark);ctx.fillStyle=C.dark;ctx.fillRect(66,244,345,20);
  text(row?.upload?'YOUR PHOTOGRAPH':row?.ood?'SPECIAL ASSISTANCE':'PASSENGER IDENTIFICATION',86,249,12,C.paleBlue,700);text(row?String(item.index+1).padStart(2,'0'):'—',390,250,15,'#fff',650,'right');
  if(row){photo(images[row.image],82,277,313,194,8);
    if(row.species){text('GOLD LABEL · REFERENCE',87,488,10,C.cobalt,750);text(row.species.toUpperCase(),86,518,27,C.ink,700);text(animalCategory(row.species),87,540,13,C.muted,600)}
    else{text('NO REFERENCE LABEL',87,488,10,C.cobalt,750);text('UNLABELLED',86,518,27,C.ink,700);text('Not counted in the score',87,540,13,C.muted,600)}
  }else{ammonite(148,305,180);text(intro?'35 animal classes':'TERMINAL READY',238,480,28,C.indigo,650,'center');text(intro?(live?'Press Start for the live run':'Press Play for the replay'):'Waiting for a measured run',238,513,15,C.muted,500,'center')}
  // A moving photo pass is the animal travelling through the checkpoint.
  if(row){const at=routedAt(item);let move=phase==='routed'&&at!=null?ease((t-at)/.7):0;const gate=GROUPS.indexOf(routedGate(row));if(gate<0)move=0;
    let px=650+(Math.max(0,gate)*209+1000-650)*move;const rise=Math.max(0,(px-890)/(Math.max(0,gate)*209+110));let py=580-123*rise;
    ctx.save();ctx.globalAlpha=1-move;rr(px-29,py-34,58,63,8,'#fff');photo(images[row.image],px-25,py-30,50,45,4);ctx.restore();
    if(phase==='escalating'){rr(452,548,177,30,7,C.rose);text('LOCAL SYSTEM TWO',540,568,10,'#ffffff',700,'center')}
    if(phase==='routed'&&!routedGate(row)){rr(452,548,177,30,7,C.rose);text('REVIEW REQUIRED',540,568,10,'#ffffff',700,'center')}
  }
  // Independent results stay separate from the photo's reference label.
  const reveal=phase!=='scanning'&&row;
  rr(38,650,450,188,16,C.blueTint,'#d8def8');text('AMMONIX',61,681,15,C.cobalt,750);text(live?'LIVE · SINGLE RUN':'MEDIAN OF 3 RUNS',158,681,10,C.muted,650);
  text(reveal?ms(timing):row&&local>=LEAD?ms(ammonixElapsed):'—',465,681,25,C.ink,650,'right');
  rr(61,694,404,5,3,'#dce1f8');
  if(row&&!reveal&&local>=LEAD){const sweep=(t*1.3)%1;rr(61+404*Math.max(0,sweep-.25),694,404*Math.min(.25,sweep),5,3,C.cobalt)}
  else if(reveal)rr(61,694,404,5,3,C.cobalt);
  text(reveal&&row.escalated?(phase==='routed'?'SYSTEM TWO · RESOLVED':'DECISION TO ESCALATE'):'STRUCTURED DECISION',61,719,10,C.muted,700);
  text(reveal?(row.escalated?(phase==='routed'?String(row.final_label).toUpperCase():'UNCERTAIN'):row.prediction.toUpperCase()):row?'SCANNING…':outro?'RUN COMPLETE':'READY TO SCAN',61,750,27,C.ink,700);
  const groupIndex=row?GROUPS.indexOf(row.gate):-1;
  const reviewed=reveal&&row.escalated&&phase==='routed'&&row.final_label;
  const reviewLabel=reviewed?`LOCAL SYSTEM TWO → ${String(row.final_label).toUpperCase()}${animalCategory(row.final_label)?' · '+animalCategory(row.final_label):''}`:'LOCAL SYSTEM TWO REVIEW';
  text(reveal?(row.escalated?reviewLabel:`${animalCategory(row.prediction)}  ·  GATE 0${groupIndex+1}`):'Image → decision',62,773,13,reviewed?C.cobalt:row?.escalated?C.rose:C.muted,600);
  text('Class score',62,798,12,C.muted);text(reveal?`${(row.confidence*100).toFixed(1)}%`:'—',465,798,18,C.ink,650,'right');
  if(reveal&&row.escalated){const status=reviewed?`FULL RESOLUTION · ${ms(row.ammonix_ms+(row.fallback_ms||0))} TOTAL`:'Uncertainty threshold triggered';text(status,62,821,11,reviewed?C.cobalt:C.rose,650)}
  else text(reveal?'Species → group → gate':'Known expertise. Fast decisions.',62,821,11,C.muted,500);
  const [rivalHead,rivalSub]=rivalTitle(row);
  rr(508,650,723,188,16,C.roseTint,'#eddbe3');text(rivalHead,531,681,15,C.rose,750);text(rivalSub,531+(rivalHead==='GPT6'?61:158),681,10,C.rose,700);
  text(answerKnown?ms(gen):rivalElapsed!=null?ms(rivalElapsed):'—',1208,681,25,C.ink,650,'right');
  rr(531,694,677,5,3,'#eddae2');
  if(answerKnown)rr(531,694,677,5,3,C.rose);
  else if(rivalElapsed!=null){const sweep=(t*1.1)%1;rr(531+677*Math.max(0,sweep-.25),694,677*Math.min(.25,sweep),5,3,C.rose)}
  text('FULL ANSWER · UNPARSED',531,719,10,C.muted,700);
  if(answerKnown&&row.answer_names_species!=null)text(row.answer_names_species?'NAMES THE REFERENCE SPECIES':'DOES NOT NAME THE REFERENCE SPECIES',1208,719,10,row.answer_names_species?C.cobalt:C.rose,650,'right');
  const waiting=live?(decisionKnown?'Generating the full answer on the same GPU…':'Waiting for the Ammonix decision before the full answer is generated.'):'Waiting for the complete generated answer. The timer shows the measured time.';
  fullAnswer(answerKnown?row.traditional_answer:row?waiting:outro?'Replay any arrival, or scan your own photograph.':'The complete generated answer will appear here.',531,728,677,96,answerKnown&&row.traditional_completion&&row.traditional_completion.complete===false?C.rose:C.ink);
  // Counters.
  rr(1251,650,311,188,16,C.dark);text('PASSENGERS PROCESSED',1274,681,12,C.paleBlue,700);
  const items=session?.items||[],decided=items.filter(i=>i.decisionAt!=null&&t>=i.decisionAt),known=decided.filter(i=>i.row.species&&!i.row.ood&&!i.row.upload),correct=known.filter(i=>i.row.correct).length;
  const answered=items.filter(i=>i.answerAt!=null&&t>=i.answerAt),named=answered.filter(i=>i.row.answer_names_species===true).length,scorable=answered.filter(i=>i.row.answer_names_species!=null).length;
  const queue=session?.queue||PASSENGERS?.rows||[],knownTotal=queue.filter(r=>r.species&&!r.ood&&!r.upload).length,unknownTotal=queue.filter(r=>r.ood).length,uploads=queue.filter(r=>r.upload).length;
  text(`${String(decided.length).padStart(2,'0')} / ${queue.length}`,1274,733,39,'#fff',600);
  text(`${knownTotal} KNOWN + ${unknownTotal} UNKNOWN${uploads?' + '+uploads+' YOURS':''} · CURATED`,1275,757,10,'#c1c7ea',700);
  text(`Ammonix top-1 correct: ${correct} / ${known.length}`,1274,785,13,C.paleBlue,600);
  text(`${rivalShort(row)} names the species: ${scorable?named+' / '+scorable:'—'}`,1274,812,15,C.paleRose,600);
  const counts=gateStats(t),boardedTotal=Object.values(counts.boarded).reduce((a,b)=>a+b,0),reviewsTotal=Object.values(counts.review).reduce((a,b)=>a+b,0);
  text(`${boardedTotal} / ${queue.length} BOARDED${reviewsTotal?' · '+reviewsTotal+' REVIEW':''}   ·   ${rivalShort(row)} ${answered.length} ANSWERED`,918,45,12,C.indigo,600,'center');
  text(live?'Live inference · every time is one measured request on this machine':'Curated cases · top-1 correctness is separate from boarding',918,65,10,C.muted,500,'center');
  text(live?`${queue.length} PASSENGERS  /  LIVE ON ${gpuShort().toUpperCase()}  /  TERNARY BONSAI 2 27B · PTQ1_0 · 1.75 BITS PER WEIGHT`:`${queue.length} PASSENGERS  /  REPLAY OF THE MEASURED RUN  /  TERNARY BONSAI 2 27B · PTQ1_0 · 1.75 BITS PER WEIGHT`,40,868,11,C.muted,650);
  text(live?'No replayed times · cache erased before every request · source credits supplied':'Measured times unchanged · original benchmark retained · source credits supplied',1560,868,11,C.muted,400,'right');
  const progress=session?Math.min(1,(session.done?queue.length:decided.length)/Math.max(1,queue.length)):0;
  const brandGradient=ctx.createLinearGradient(0,0,1600,0);brandGradient.addColorStop(0,C.indigo);brandGradient.addColorStop(1,C.cobalt);ctx.fillStyle=brandGradient;ctx.fillRect(0,895,1600*progress,5);
  if(intro){text('01',1471,142,33,C.cobalt,700,'right')}
  if(outro){ctx.save();ctx.fillStyle='#f5f7fcf2';ctx.fillRect(38,192,1524,437);
    const ammonixTimes=known.map(i=>i.row.ammonix_ms),rivalTimes=answered.map(i=>i.row.traditional_ms);
    text(live?`LIVE RUN COMPLETE · ${queue.length} PASSENGERS ON ${gpuShort().toUpperCase()}`:`REPLAY COMPLETE · ${queue.length} PASSENGERS · MEASURED ON ${String(REPLAY?.gpu||'').toUpperCase()}`,800,239,13,C.indigo,750,'center');
    rr(206,261,571,184,16,C.blueTint,'#d8def8');rr(823,261,571,184,16,C.roseTint,'#eddbe3');
    text('AMMONIX',491,291,17,C.cobalt,750,'center');text(rivalOf(row)==='gpt6'?'GPT6 · MAX REASONING':'TRADITIONAL LLM · SAME MODEL',1108,291,17,C.rose,750,'center');
    text(known.length?`${(100*correct/known.length).toFixed(1)}%`:'—',491,352,53,C.indigo,650,'center');text(scorable?`${(100*named/scorable).toFixed(1)}%`:'—',1108,352,53,C.indigo,650,'center');
    text(`${correct} / ${known.length} top-1 correct`,491,381,17,C.muted,500,'center');text(scorable?`${named} / ${scorable} answers name the species`:'answers not scored',1108,381,17,C.muted,500,'center');
    text(`MEDIAN DECISION TIME   ${ms(median(ammonixTimes))}`,491,422,21,C.cobalt,650,'center');text(`MEDIAN FULL ANSWER   ${ms(median(rivalTimes))}`,1108,422,21,C.rose,650,'center');
    text(live?'Single live measurements on this machine · both paths on the same 1.75-bit weights · cache erased before every request':'Medians of three trials per photograph · both paths on the same 1.75-bit weights',800,472,14,C.muted,500,'center');
    text('Known expertise → System One. Uncertainty → System Two.',800,527,25,C.ink,600,'center');
    text(`BOARDED: ${counts.boarded['MAMMALS']} mammals · ${counts.boarded['BIRDS & REPTILES']} birds/reptiles · ${counts.boarded['ARTHROPODS']} arthropods · ${items.filter(i=>i.row.escalated).length} System Two resolutions${reviewsTotal?' · '+reviewsTotal+' review':''}`,800,575,14,C.indigo,650,'center');
    text('Curated passengers: a demonstration, not a population sample. The 70-photograph benchmark is in the repository.',800,607,12,C.muted,400,'center');ctx.restore()}
  $('status').textContent=session?`${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}`+(session.mode==='replay'?` / ${Math.floor(session.duration/60)}:${String(Math.floor(session.duration%60)).padStart(2,'0')}`:''):'0:00';
  if(session?.mode==='replay')$('seek').value=1000*t/Math.max(.001,session.duration);
}

// ---- session control ---------------------------------------------------------------------------------------------------
const sleep=msec=>new Promise(r=>setTimeout(r,msec));
async function api(path,body){const res=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await res.json();if(!res.ok||data.error)throw Error(data.error||res.statusText);return data}
function loadImage(src){if(images[src])return images[src];const im=new Image();im.src=src;images[src]=im;return im}
function freshRows(){return (PASSENGERS?.rows||[]).map(r=>({id:r.id,species:r.species,image:r.image,gold_gate:r.gate,ood:!!r.ood,split:r.split,author:r.author,license:r.license}))}
function newSession(mode){
  session={mode,queue:freshRows(),items:[],elapsed:0,paused:false,done:false,running:false,duration:1,startedAt:performance.now()};
  if(mode==='replay')scheduleReplay();
  return session;
}
// Live sessions run on the wall clock, so the recorded moments stay right even while the tab is hidden.
const clockOf=s=>(performance.now()-s.startedAt)/1000;
function scheduleReplay(){
  // The measured rows of ternary/results.json for every passenger that has one, at real speed; both paths from the same start as in the film.
  session.queue=session.queue.filter(r=>REPLAY?.rows?.[r.id]);
  let t=4;
  session.queue.forEach((row,index)=>{
    const m=REPLAY.rows[row.id];
    Object.assign(row,{prediction:m.prediction,gate:m.gate,confidence:m.confidence,novelty_score:m.novelty_score,escalated:!!m.escalated,final_label:m.final_label,final_gate:m.final_gate,fallback_ms:m.fallback_ms||0,ammonix_ms:m.ammonix_ms,probabilities:m.probabilities||null,traditional_answer:m.traditional_answer,traditional_ms:m.traditional_ms,traditional_completion:m.traditional_completion,rival:'local',correct:m.correct,answer_names_species:null});
    const item={row,index,start:t,decisionAt:t+LEAD+m.ammonix_ms/1000,answerAt:t+LEAD+m.traditional_ms/1000,end:null};
    item.end=Math.max(routedAt(item),item.answerAt)+HOLD;
    session.items.push(item);t=item.end;
  });
  session.duration=t+7;
  session.running=true;
}
async function runLive(mySession){
  mySession.running=true;
  try{
    while(session===mySession){
      if(mySession.paused){await sleep(120);continue}
      const index=mySession.items.length;
      if(index>=mySession.queue.length)break;
      const row=mySession.queue[index];
      const item={row,index,start:clockOf(mySession),decisionAt:null,answerAt:null,end:null};
      mySession.items.push(item);
      await sleep(LEAD*1000);
      const body=row.upload?{image:row.b64}:{id:row.id};
      try{
        const decision=await api('/api/decide',body);
        if(session!==mySession)return;
        Object.assign(row,decision);
        item.decisionAt=clockOf(mySession);
        const answer=await api('/api/answer',{...body,rival:rivalOf(null)});
        if(session!==mySession)return;
        Object.assign(row,answer);
        item.answerAt=clockOf(mySession);
      }catch(e){
        if(session!==mySession)return;
        if(item.decisionAt==null){Object.assign(row,{prediction:'error',gate:null,confidence:0,escalated:true,final_label:'Review required',final_gate:null,fallback_ms:0,ammonix_ms:0,probabilities:null,correct:null});item.decisionAt=clockOf(mySession)}
        Object.assign(row,{traditional_answer:'The request failed: '+e.message,traditional_ms:null,traditional_completion:{complete:false},answer_names_species:null});item.answerAt=clockOf(mySession);
        $('mode').textContent='Live request failed: '+e.message;
      }
      const hold=HOLD+Math.max(0,(routedAt(item)+.7)-clockOf(mySession));
      await sleep(hold*1000);
      item.end=clockOf(mySession);
    }
    if(session===mySession){mySession.done=true;updateControls()}
  }finally{if(session===mySession)mySession.running=false}
}
function updateControls(){
  const live=STATUS?.live,play=$('play');
  if(!STATUS){play.disabled=true;return}
  if(live){
    play.disabled=false;
    play.textContent=!session?'Start live run':session.done?'Run again':session.paused?'Resume':'Pause';
    $('restart').disabled=!session;
    $('seek').disabled=true;$('seek').value=session?1000*Math.min(1,session.items.length/Math.max(1,session.queue.length)):0;
    $('uploadLabel').classList.toggle('disabled',false);
  }else{
    const ok=!!REPLAY;
    play.disabled=!ok;
    play.textContent=!ok?'No measured run':!session?'Play replay':session.done?'Replay again':session.paused?'Play':'Pause';
    $('restart').disabled=!session;$('seek').disabled=!session;
    $('uploadLabel').classList.add('disabled');
  }
}
function tick(now){
  const dt=lastTick==null?0:Math.min(.25,(now-lastTick)/1000);lastTick=now;
  if(session){
    if(session.mode==='live')session.elapsed=clockOf(session);
    else if(!session.paused&&session.running){session.elapsed+=dt;if(session.elapsed>=session.duration){session.elapsed=session.duration;session.done=true;session.running=false;updateControls()}}
    renderFrame(session.elapsed);
  }else renderFrame(0);
  requestAnimationFrame(tick);
}
$('play').onclick=()=>{
  if(!STATUS)return;
  if(STATUS.live){
    if(!session||session.done){newSession('live');runLive(session)}
    else session.paused=!session.paused;
  }else{
    if(!REPLAY)return;
    if(!session||session.done){newSession('replay')}
    else session.paused=!session.paused;
  }
  updateControls();
};
$('restart').onclick=()=>{if(!STATUS)return;if(STATUS.live){newSession('live');runLive(session)}else if(REPLAY){newSession('replay')}updateControls()};
$('seek').oninput=e=>{if(session?.mode==='replay'){session.paused=true;session.elapsed=e.target.value*session.duration/1000;updateControls();}};
$('upload').onchange=async e=>{
  const file=e.target.files?.[0];e.target.value='';
  if(!file||!STATUS?.live)return;
  const bitmap=await createImageBitmap(file).catch(()=>null);
  if(!bitmap){$('mode').textContent='That file could not be read as an image.';return}
  const scale=Math.min(1,1024/Math.max(bitmap.width,bitmap.height)),w=Math.round(bitmap.width*scale),h=Math.round(bitmap.height*scale);
  const off=document.createElement('canvas');off.width=w;off.height=h;off.getContext('2d').drawImage(bitmap,0,0,w,h);
  const dataUrl=off.toDataURL('image/jpeg',.92);
  const row={id:'upload-'+Date.now(),species:null,image:dataUrl,gold_gate:null,ood:false,upload:true,b64:dataUrl};
  loadImage(dataUrl);
  if(!session||session.done){newSession('live');session.queue=[row];runLive(session)}
  else session.queue.splice(session.items.length+(session.items.at(-1)?.end==null?1:0),0,row);
  $('mode').textContent=`Your photograph is queued as passenger ${session.queue.indexOf(row)+1}.`;
  updateControls();
};
async function init(){
  try{
    await brandReady;
    STATUS=await (await fetch('/api/status',{cache:'no-store'})).json();
    PASSENGERS=await (await fetch('/api/passengers',{cache:'no-store'})).json();
    for(const r of PASSENGERS.rows)loadImage(r.image);
    if(!STATUS.live){const res=await fetch('/api/replay',{cache:'no-store'});REPLAY=res.ok?await res.json():null}
    $('rival').hidden=!STATUS.gpt6;
    $('mode').textContent=STATUS.live?`Live · ${STATUS.model} · ${STATUS.gpu||'GPU'} · llama.cpp ${STATUS.server||''} · ${PASSENGERS.rows.length} curated passengers, or scan your own photograph`:REPLAY?`Replay · ${REPLAY.model} · measured on ${REPLAY.gpu} · ${PASSENGERS.rows.filter(r=>REPLAY.rows[r.id]).length} of ${PASSENGERS.rows.length} curated passengers have measured rows`:'No measured run found (ternary/results.json) and no live model. Start with serve_ternary.py --live.';
  }catch(e){$('mode').textContent='The terminal could not be reached: '+e.message}
  updateControls();
  requestAnimationFrame(tick);
}
window.renderFrame=renderFrame;
init();
