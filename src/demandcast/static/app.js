const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const state = {config:null, benchmark:null, forecast:null, explanationSequence:0, forecastSequence:0};
const labels = {lag_24:'Same hour yesterday',lag_48:'Same hour two days ago',lag_168:'Same hour last week',rolling_mean_24:'Recent daily demand level',rolling_mean_168:'Recent weekly demand level',hour_sin:'Hour of day · sine',hour_cos:'Hour of day · cosine',weekday_sin:'Day of week · sine',weekday_cos:'Day of week · cosine',month_sin:'Month · sine',month_cos:'Month · cosine',is_weekend:'Weekend indicator',day_index:'Time trend'};
const format = (number, digits=0) => Number(number).toLocaleString('en-US',{maximumFractionDigits:digits,minimumFractionDigits:digits});
const percent = number => `${(number*100).toFixed(1)}%`;
const hour = number => `${String(number).padStart(2,'0')}:00`;
function element(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
function svgElement(tag,attributes,text){const node=document.createElementNS(NS,tag);for(const [key,value] of Object.entries(attributes))node.setAttribute(key,String(value));if(text!==undefined)node.textContent=text;return node;}
async function api(path,options={}){
  const headers=new Headers(options.headers||{});
  if($('token').value)headers.set('Authorization',`Bearer ${$('token').value}`);
  if(options.body)headers.set('Content-Type','application/json');
  const response=await fetch(path,{...options,headers});
  const data=await response.json();
  if(!response.ok){if(response.status===401)$('auth-panel').hidden=false;throw new Error(typeof data.detail==='string'?data.detail:'Check the request and try again.');}
  return data;
}
async function run(action){
  $('notice').textContent='';
  try{await action();}catch(error){$('notice').textContent=error.message;}
}
function switchTab(name){
  for(const tab of ['forecast','benchmark','monitor'])$(tab+'-tab').hidden=tab!==name;
  document.querySelectorAll('[data-tab]').forEach(button=>button.classList.toggle('active',button.dataset.tab===name));
  $('page-title').textContent={forecast:'Forecast overview',benchmark:'Model benchmark',monitor:'Drift monitor'}[name];
}
document.querySelectorAll('[data-tab]').forEach(button=>button.onclick=()=>switchTab(button.dataset.tab));

function renderChart(){
  const result=state.forecast;if(!result)return;
  const points=result.points;
  const capacity=Number($('capacity').value);
  $('capacity-value').textContent=`${format(capacity)} / h`;
  const svg=$('forecast-chart');svg.replaceChildren();
  const left=43,right=881,top=17,bottom=254;
  const maxValue=Math.ceil(Math.max(capacity,...points.map(point=>Math.max(point.upper,point.actual||0)))*1.12/100)*100;
  const x=index=>left+index*(right-left)/Math.max(points.length-1,1);
  const y=value=>bottom-value/maxValue*(bottom-top);
  for(let tick=0;tick<=4;tick++){
    const value=maxValue*tick/4;
    svg.append(svgElement('line',{x1:left,x2:right,y1:y(value),y2:y(value),stroke:'#edf1f7','stroke-width':1}));
    svg.append(svgElement('text',{x:left-10,y:y(value)+3,fill:'#a7b2c4','font-size':9,'text-anchor':'end'},format(value)));
  }
  points.forEach((point,index)=>{if(index%3===0||index===points.length-1)svg.append(svgElement('text',{x:x(index),y:bottom+22,fill:'#a7b2c4','font-size':9,'text-anchor':'middle'},hour(point.hour)));});
  const area=points.map((point,index)=>`${x(index)},${y(point.upper)}`).concat([...points].reverse().map((point,index)=>`${x(points.length-1-index)},${y(point.lower)}`));
  svg.append(svgElement('polygon',{points:area.join(' '),fill:'#e8efff',opacity:0.85}));
  svg.append(svgElement('line',{x1:left,x2:right,y1:y(capacity),y2:y(capacity),stroke:'#d6ac70','stroke-dasharray':'5 5','stroke-width':1}));
  function line(key,color,width){
    let path='',inSegment=false;
    points.forEach((point,index)=>{if(point[key]===null){inSegment=false;return;}path+=`${inSegment?'L':'M'}${x(index)},${y(point[key])} `;inSegment=true;});
    return svgElement('path',{d:path,fill:'none',stroke:color,'stroke-width':width,'stroke-linejoin':'round','stroke-linecap':'round'});
  }
  svg.append(line('actual','#52647e',1.5),line('predicted','#3a6aee',2.5));
  points.forEach((point,index)=>{
    const dot=svgElement('circle',{cx:x(index),cy:y(point.predicted),r:2.7,fill:'#fff',stroke:'#3a6aee','stroke-width':1.5});
    dot.append(svgElement('title',{},`${hour(point.hour)}: forecast ${format(point.predicted,1)}, observed ${point.actual===null?'unavailable':format(point.actual)}, interval ${format(point.lower,1)}–${format(point.upper,1)}`));
    svg.append(dot);
  });
  const count=points.filter(point=>point.predicted>capacity).length;
  $('capacity-summary').textContent=`${count} forecast hour${count===1?'':'s'} above the illustrative service capacity`;
  function pointForEvent(event){const rect=svg.getBoundingClientRect();const relative=(event.clientX-rect.left)/rect.width*900;const index=Math.max(0,Math.min(points.length-1,Math.round((relative-left)/(right-left)*(points.length-1))));return points[index];}
  svg.onmousemove=event=>{const point=pointForEvent(event);$('chart-tooltip').textContent=`${hour(point.hour)} · forecast ${format(point.predicted,1)} · observed ${point.actual===null?'unavailable':format(point.actual)} · interval ${format(point.lower,1)}–${format(point.upper,1)} rentals/h`;};
  svg.onclick=event=>{const point=pointForEvent(event);$('explain-hour').value=point.timestamp;run(loadExplanation);};
}
function renderForecast(result){
  state.forecast=result;
  $('total-demand').textContent=format(result.total_predicted_rentals);
  $('peak-demand').textContent=format(Math.max(...result.points.map(point=>point.predicted)));
  $('peak-hour').textContent=`Expected at ${hour(result.peak_hour)} · rentals/hour`;
  $('forecast-origin').textContent=`Forecast origin: ${result.forecast_origin.replace('T',' ')} · dataset-local time`;
  $('latency').textContent=`${format(result.latency_ms,1)} ms · verified local model`;
  $('explain-hour').replaceChildren(...result.points.map(point=>{const option=element('option',hour(point.hour));option.value=point.timestamp;return option;}));
  const defaultHour=result.points.find(point=>point.hour===8)||result.points[0];
  $('explain-hour').value=defaultHour.timestamp;
  $('hourly-rows').replaceChildren(...result.points.map(point=>{const row=element('tr');for(const value of [hour(point.hour),format(point.predicted,1),point.actual===null?'—':format(point.actual),format(point.lower,1),format(point.upper,1)])row.append(element('td',value));return row;}));
  renderChart();
}
async function loadForecast(){
  const sequence=++state.forecastSequence;
  $('forecast-date').disabled=true;
  try{const result=await api(`/api/forecast?date=${encodeURIComponent($('forecast-date').value)}`);if(sequence!==state.forecastSequence)return;renderForecast(result);await loadExplanation();}
  finally{if(sequence===state.forecastSequence)$('forecast-date').disabled=false;}
}
async function loadExplanation(){
  const sequence=++state.explanationSequence;
  const result=await api(`/api/explain?timestamp=${encodeURIComponent($('explain-hour').value)}`);
  if(sequence!==state.explanationSequence)return;
  $('explain-prediction').textContent=format(result.display_prediction);
  $('explain-base').textContent=`${format(result.base_value,1)} rentals/h`;
  const contributions=result.contributions.slice(0,6);
  const maximum=Math.max(...contributions.map(row=>Math.abs(row.contribution)),1);
  $('contributions').replaceChildren(...contributions.map(row=>{
    const card=element('div','','contribution');const title=element('div','','contribution-title');
    title.append(element('span',labels[row.feature]||row.feature),element('span',`${row.contribution>=0?'+':''}${format(row.contribution,1)}`,`contribution-value ${row.contribution>=0?'positive':'negative'}`));
    const track=element('div','','bar-track');const bar=element('div','',`bar ${row.contribution<0?'negative':''}`);bar.style.width=`${Math.abs(row.contribution)/maximum*100}%`;track.append(bar);card.append(title,track);return card;
  }));
}
function renderBenchmark(report){
  $('test-mae').textContent=format(report.selected_test_metrics.mae,1);
  $('coverage').textContent=percent(report.intervals.coverage);
  $('holdout-size').textContent=`${format(report.selected_test_metrics.rows)} held-out target hours`;
  const selected=report.selection.selected_candidate;
  $('leaderboard').replaceChildren(...report.leaderboard.map(item=>{
    const row=element('tr','',item.name===selected?'selected':'');const name=element('td',item.name.replaceAll('_',' '));
    if(item.name===selected)name.append(element('small','SELECTED CANDIDATE'));
    row.append(name,element('td',format(item.mae,2)),element('td',format(item.rmse,2)),element('td',percent(item.wape)));return row;
  }));
  const gain=report.mae_improvement_vs_weekly_baseline_percent;
  $('benchmark-summary').textContent=`The selected CatBoost model has ${Math.abs(gain).toFixed(1)}% ${gain>=0?'lower':'higher'} MAE than the weekly seasonal baseline on this historical test. This is not a production-performance claim.`;
  $('worst-days').replaceChildren(...report.worst_days.slice(0,5).map(day=>{const row=element('div','','worst-day');row.append(element('span',day.date),element('strong',`${format(day.mae,1)} rentals/h MAE`));return row;}));
  const registry=report.registry;
  const rows=[['Selected model',selected],['MLflow registry',registry.enabled?`${registry.name} · version ${registry.version}`:'Tracking disabled for this run'],['Registry alias',registry.alias||'—'],['Selection used test data','No'],['Training observations',format(report.splits.train.rows)],['Missing clock hours',format(report.missing_clock_hours)],['Dataset version',report.source.archive_sha256.slice(0,16)+'…']];
  $('lineage').replaceChildren(...rows.map(([key,value])=>{const row=element('div');row.append(element('dt',key),element('dd',value));return row;}));
}
function renderMonitor(report){
  const simulated=report.simulated;
  $('monitor-status').textContent=`${simulated?'SIMULATION · ':''}${report.status.replaceAll('_',' ').toUpperCase()}`;
  $('monitor-headline').textContent=report.status==='insufficient_sample'?'More observations are needed for this diagnostic':report.status==='review'?'Feature distributions differ from the reference':'No large marginal shift under this heuristic';
  $('monitor-period').textContent=`${report.rows} hourly rows · reference ${report.reference_period.start.slice(0,10)} to ${report.reference_period.end.slice(0,10)}${simulated?` · history inputs scaled ${report.history_scale_factor.toFixed(2)}×`:''}`;
  const maximum=Math.max(...report.features.map(row=>row.psi),.4);
  $('drift-features').replaceChildren(...report.features.map(row=>{const card=element('div','','drift-row');const title=element('div','','drift-title');title.append(element('span',labels[row.feature]||row.feature),element('strong',row.psi.toFixed(3)));const track=element('div','','bar-track');const bar=element('div','',`bar ${row.signal==='review'?'review':''}`);bar.style.width=`${row.psi/maximum*100}%`;track.append(bar);card.append(title,track,element('div',`Missing values: ${percent(row.missing_rate)} · ${row.signal==='review'?'review suggested':'below review threshold'}`,'drift-meta'));return card;}));
}
async function initialize(){
  const [config,benchmark,monitor]=await Promise.all([api('/api/config'),api('/api/benchmark'),api('/api/monitor')]);
  state.config=config;state.benchmark=benchmark;
  $('auth-panel').hidden=true;
  $('candidate').textContent=`Candidate model · ${config.model.replaceAll('_',' ')}`;
  $('forecast-date').replaceChildren(...config.dates.map(date=>{const option=element('option',new Intl.DateTimeFormat('en-GB',{day:'2-digit',month:'short',year:'numeric',timeZone:'UTC'}).format(new Date(date+'T12:00:00Z')));option.value=date;return option;}));
  $('forecast-date').value=config.default_date;
  renderBenchmark(benchmark);renderMonitor(monitor);await loadForecast();
}
$('forecast-date').onchange=()=>run(loadForecast);
$('explain-hour').onchange=()=>run(loadExplanation);
$('capacity').oninput=renderChart;
$('shift-factor').oninput=()=>{$('shift-label').textContent=Number($('shift-factor').value).toFixed(2)+'×';};
$('run-shift').onclick=()=>run(async()=>{const button=$('run-shift');button.disabled=true;try{renderMonitor(await api('/api/monitor/simulate',{method:'POST',body:JSON.stringify({factor:Number($('shift-factor').value)})}));}finally{button.disabled=false;}});
$('reset-monitor').onclick=()=>run(async()=>renderMonitor(await api('/api/monitor')));
$('unlock').onclick=()=>run(initialize);
$('export').onclick=()=>{if(!state.forecast)return;const url=URL.createObjectURL(new Blob([JSON.stringify(state.forecast,null,2)],{type:'application/json'}));const anchor=element('a');anchor.href=url;anchor.download=`DemandCast_forecast_${state.forecast.date}.json`;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
run(initialize);
