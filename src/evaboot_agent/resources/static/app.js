const state = {runId:null, run:null, workspace:null, health:null, view:'overview', newRun:false};
const $ = (selector, root=document) => root.querySelector(selector);
const $$ = (selector, root=document) => [...root.querySelectorAll(selector)];
const el = (tag, className, text) => { const node=document.createElement(tag); if(className)node.className=className; if(text!==undefined)node.textContent=text; return node; };
const safe = value => value == null ? '' : String(value);

async function api(path, options={}) {
  const response = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
  const payload = await response.json().catch(()=>({detail:'The server returned an unreadable response.'}));
  if(!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}
function toast(message){const box=$('#toast');box.textContent=message;box.classList.add('show');setTimeout(()=>box.classList.remove('show'),3300);}
function fmtTime(value){if(!value)return '';const date=new Date(value);return Number.isNaN(date.valueOf())?'':date.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});}
function fmtDate(value){if(!value)return '';const date=new Date(value);return Number.isNaN(date.valueOf())?'':date.toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});}
function runTitle(run){return safe(run?.goal?.goal).slice(0,35) || run?.id?.slice(0,8) || 'New run';}
function setView(name){state.view=name;$$('.view').forEach(v=>v.classList.toggle('active',v.id===`view-${name}`));$$('.nav-tab').forEach(b=>b.classList.toggle('active',b.dataset.view===name));$('#breadcrumb-current').textContent={overview:'Overview',prospects:'Prospects',trace:'Run trace',learning:'Learning'}[name]||name;}
function statusClass(status){if(['complete','human_approved','auto_promoted'].includes(status))return 'pill-green';if(['running','awaiting_human_review','awaiting_human_approval','awaiting_canary'].includes(status))return 'pill-blue';if(['deferred','budget_exhausted','rejected_by_gate','rejected_by_canary','rejected_canary_incomparable','superseded','human_rejected'].includes(status))return 'pill-amber';if(['failed','rejected','skipped_insufficient_evidence'].includes(status))return 'pill-red';return 'pill-neutral';}
function backendReady(backend=$('#agent-backend')?.value){if(backend==='codex_subscription')return !!state.health?.codex_subscription_available;return !!state.health?.openai_configured;}
function runBackend(run){return run?.goal?.agent_backend||'openai_api';}

async function refreshWorkspace(){
  try{
    const data=await api('/api/state');state.workspace=data.workspace;state.health=data.health;
    renderKeys();renderRunList();renderLearning();
    if(state.runId){const run=await api(`/api/runs/${encodeURIComponent(state.runId)}`);renderRun(run);}
    else if(state.workspace.runs.length&&!state.newRun){await selectRun(state.workspace.runs[0].id,false);}
  }catch(error){console.error(error);}
}
function renderKeys(){
  const evaboot=$('#data-source').value==='evaboot';$('#live-auth-row').classList.toggle('hidden',!evaboot);
  $('#live-auth').disabled=!state.health?.evaboot_configured;
  if(evaboot&&!state.health?.evaboot_configured)$('#live-auth-row').querySelector('p').textContent='EVABOOT_API_KEY is not configured for this process.';
}
function renderRunList(){
  const container=$('#run-list');container.replaceChildren();const runs=state.workspace?.runs||[];
  if(!runs.length){container.append(el('div','empty-sidebar','No runs yet'));return;}
  for(const run of runs){const button=el('button','run-item');button.dataset.status=run.status;button.classList.toggle('selected',run.id===state.runId);button.append(el('span','run-dot'));button.append(el('span','run-item-name',runTitle({goal:{goal:run.id===state.runId?state.run?.goal?.goal:run.id}})));button.title=run.goal||run.id;button.querySelector('.run-item-name').textContent=run.id===state.runId?runTitle(state.run):run.id.slice(0,13);button.addEventListener('click',()=>selectRun(run.id));container.append(button);}
}
async function selectRun(runId, switchToOverview=false){state.runId=runId;state.newRun=false;try{renderRun(await api(`/api/runs/${encodeURIComponent(runId)}`));if(switchToOverview)setView('overview');}catch(error){toast(error.message);}}

function renderRun(run){
  state.run=run;$('#activity-title').textContent=runTitle(run);const badge=$('#run-status');badge.textContent=run.status.replaceAll('_',' ');badge.className=`pill ${statusClass(run.status)}`;
  $('#activity-empty').classList.toggle('hidden',!!run.events?.length);
  const counters=run.counters||{};const metrics=$('#activity-metrics');metrics.replaceChildren();
  const subscription=runBackend(run)==='codex_subscription';
  const usageValue=subscription?`~$${Number(counters.cost_usd||0).toFixed(4)} equiv.`:`$${Number(counters.cost_usd||0).toFixed(4)}`;
  [["TOOL CALLS",counters.tool_calls||0],["LEADS REVIEWED",run.leads?.length||0],["LUNA USAGE EST.",usageValue],["POLICY VERSION",run.policy_version||'—']].forEach(([label,value])=>{const box=el('div');box.append(el('small','',label),el('strong','',safe(value)));metrics.append(box);});
  $('#plan-placeholder').classList.toggle('hidden',!!run.plan);$('#plan-content').classList.toggle('hidden',!run.plan);$('#plan-actions').classList.toggle('hidden',!run.plan);
  if(run.plan)renderPlan(run.plan,run);
  const stop=$('#stop-run');stop.classList.toggle('hidden',!['running','planned','deferred','awaiting_human_review'].includes(run.status));
  $('#prospect-count').textContent=run.leads?.length||0;$('#prospect-run-label').textContent=run.id.slice(0,8);$('#trace-run-label').textContent=run.id.slice(0,8);
  renderEvents(run.events||[]);renderProspects(run.leads||[]);renderRunList();
  $('#run-improvement').disabled=!backendReady(runBackend(run))||!state.health?.typesafe_configured||!state.runId;
  const executeButton=$('#execute-run');executeButton.disabled=!backendReady(runBackend(run))||!state.health?.typesafe_configured||!['planned','deferred'].includes(run.status);
  if(run.status==='running')executeButton.textContent='Run in progress…';else executeButton.innerHTML='Approve plan &amp; run <span>→</span>';
}
function renderPlan(plan,run){
  const root=$('#plan-content');root.replaceChildren();root.append(el('p','plan-summary',plan.summary));
  const chunks=[['SEARCH STRATEGY',[plan.search_prompt]],['TARGET CRITERIA',plan.target_criteria],['EXCLUSIONS',plan.exclusions],['EVIDENCE REQUIREMENTS',plan.evidence_requirements],['STOPPING CONDITIONS',plan.stopping_conditions]];
  chunks.forEach(([title,items])=>{if(!items?.length)return;const block=el('div','plan-block');block.append(el('h3','',title));const list=el('ul');items.forEach(text=>list.append(el('li','',text)));block.append(list);root.append(block);});
  const meta=el('div','plan-meta');[["LEAD CAP",plan.max_leads],["TOOL CALLS",plan.max_tool_calls],["REVISIONS",plan.max_search_revisions]].forEach(([label,value])=>{const box=el('div');box.append(el('small','',label),el('strong','',safe(value)));meta.append(box);});root.append(meta);
}
function renderEvents(events){
  const short=[...events].slice(-8).reverse();const list=$('#overview-events');list.replaceChildren();
  for(const event of short){const row=el('div','event-row');row.append(el('time','event-time',fmtTime(event.occurred_at)));const marker=el('span',`event-marker ${event.error_status?'fail':event.step.includes('budget')?'warn':''}`);row.append(marker);const copy=el('div','event-copy');copy.append(el('strong','',event.step.replaceAll('.',' / ')),el('p','',event.summary));row.append(copy);if(event.model)row.append(el('span','event-tag',`${event.model}${runBackend(state.run)==='codex_subscription'?' · plan':''}${event.input_tokens?` · ${event.input_tokens+event.output_tokens} est. tokens`:''}`));list.append(row);}
  const trace=$('#trace-list');trace.replaceChildren();if(!events.length){trace.append(emptyPanel('⌁','Trace is empty','Select or start a run.'));return;}
  for(const event of events){const row=el('div','trace-row');row.append(el('time','trace-date',fmtDate(event.occurred_at)),el('div','trace-step',event.step),el('div','trace-summary',event.summary));const meta=[];if(event.model)meta.push(event.model);if(event.input_tokens||event.output_tokens)meta.push(`${(event.input_tokens||0)+(event.output_tokens||0)} tokens`);if(event.cost_usd)meta.push(`$${Number(event.cost_usd).toFixed(5)}`);if(event.error_status)meta.push(event.error_status);row.append(el('div','trace-meta',meta.join(' · ')));trace.append(row);}
}
function emptyPanel(icon,title,description){const panel=el('div','empty-large');panel.append(el('div','placeholder-icon',''+icon),el('strong','',title),el('p','',description));return panel;}
function renderProspects(leads){
  const root=$('#prospect-grid');root.replaceChildren();if(!leads.length){root.append(emptyPanel('◎','No prospect decisions yet','Choose a run or create a new plan to inspect the evidence and routing decisions.'));return;}
  for(const lead of leads){const card=el('article','card prospect-card');const top=el('div','prospect-top');const identity=el('div');identity.append(el('h3','prospect-name',lead.name||lead.lead_id),el('div','prospect-role',lead.current_job||'Current role unavailable'),el('div','prospect-company',[lead.company,lead.company_industry].filter(Boolean).join(' · ')));top.append(identity,el('span',`pill ${statusClass(lead.status)}`,lead.status.replaceAll('_',' ')));card.append(top);
    if(lead.reason)card.append(el('div','prospect-reason',lead.reason.replaceAll('_',' ')));
    const evidence=lead.draft?.evidence||[];if(evidence.length){card.append(el('div','evidence-title','DRAFT EVIDENCE'));const list=el('div','evidence-list');for(const ref of evidence){const item=el('div','evidence-item');item.append(el('strong','',ref.field),el('span','',` — “${ref.excerpt}”`));list.append(item);}card.append(list);}
    const answers=lead.judgments?.answers||{};const judgeRows=Object.entries(answers).map(([key,value])=>[key,value.probability_yes]);if(judgeRows.length){card.append(el('div','evidence-title','JEV CHECKS'));for(const [key,p] of judgeRows){if(p===undefined)continue;const line=el('div','evidence-item');line.append(el('strong','',key.replaceAll('_',' ')),el('span','',` — ${(Number(p)*100).toFixed(0)}% yes`));card.append(line);}}
    if(lead.draft?.body){const draft=el('div','draft-box');draft.textContent=`${lead.draft.subject}\n\n${lead.draft.body}`;card.append(draft);}
    if(lead.status==='awaiting_human_review'){const actions=el('div','review-actions');const approve=el('button','button button-green','Approve sandbox');approve.addEventListener('click',()=>reviewLead(lead.lead_id,'approve'));const reject=el('button','button button-outline','Reject');reject.addEventListener('click',()=>{const reason=prompt('Why should this recommendation be rejected?');if(reason)reviewLead(lead.lead_id,'reject',reason);});const edit=el('button','button button-outline','Revise');edit.addEventListener('click',()=>{const reason=prompt('What correction should the agent use?');if(reason)reviewLead(lead.lead_id,'edit',reason);});actions.append(approve,reject,edit);card.append(actions);}
    root.append(card);
  }
}
async function reviewLead(leadId,decision,reason=''){try{const run=await api(`/api/runs/${encodeURIComponent(state.runId)}/leads/${encodeURIComponent(leadId)}/review`,{method:'POST',body:JSON.stringify({decision,reason})});renderRun(run);toast(`Review recorded: ${decision}`);}catch(error){toast(error.message);}}
function renderLearning(){
  const workspace=state.workspace;if(!workspace)return;
  const policy=workspace.policy||{};$('#active-policy-version').textContent=policy.version||'—';const detail=$('#policy-detail');detail.replaceChildren();
  const thresholds=policy.question_thresholds||{};Object.entries(thresholds).forEach(([key,value])=>{const row=el('div');row.append(el('span','',key.replaceAll('_',' ')+': '),el('code','',`${Number(value)*100}%`));detail.append(row);});
  const outcome=$('#outcome-detail');outcome.replaceChildren();const outcomes=workspace.outcomes||[];if(!outcomes.length)outcome.append(el('div','empty-inline','No outcome events stored.'));for(const item of outcomes){const row=el('div','outcome-row');row.append(el('span','',`${item.event_type}${item.reply_class?` · ${item.reply_class}`:''}`),el('strong','',safe(item.n)));outcome.append(row);}
  const candidates=$('#candidate-list');candidates.replaceChildren();const items=workspace.candidates||[];if(!items.length)candidates.append(el('div','empty-inline','No candidate changes yet.'));for(const candidate of items){const card=el('div','candidate-item');const top=el('div','candidate-item-top'),copy=el('div');copy.append(el('h3','',candidate.proposal?.change_type?.replaceAll('_',' ')||'Policy candidate'),el('p','',candidate.proposal?.rationale||''),el('p','',candidate.proposal?.expected_tradeoff||''));top.append(copy,el('span',`pill ${statusClass(candidate.status)}`,candidate.status.replaceAll('_',' ')));card.append(top);const q=candidate.evaluation?.candidate?.qualification;const base=candidate.evaluation?.baseline?.qualification;if(q&&base){card.append(el('div','candidate-metrics',`Held out ${candidate.evaluation.heldout_case_count||q.sample_count}: precision ${pct(q.precision_among_auto_approved)} vs ${pct(base.precision_among_auto_approved)} baseline · coverage ${pct(q.automatic_action_coverage)} vs ${pct(base.automatic_action_coverage)} · false approvals ${q.false_auto_approved}/${q.negative_case_denominator}`));}
      const calibrated=candidate.evaluation?.threshold_calibration?.thresholds;const replyFloor=candidate.evaluation?.calibration?.reply_threshold_calibration?.selected_threshold;if(calibrated)card.append(el('p','',`Calibration thresholds (Python): role ${pct(calibrated.current_role_fit)}, company ${pct(calibrated.company_fit)}, claim ${pct(calibrated.claim_support)}, contradiction risk ${pct(calibrated.contradiction_risk_max)}, reply clear floor ${pct(replyFloor)}.`));
      const canary=candidate.canary;if(canary?.observed_events)card.append(el('p','',`Shadow canary: ${canary.candidate_observations} candidate observations, adverse ${pct(canary.candidate_adverse_rate)} vs ${pct(canary.baseline_adverse_rate)} baseline.`));
      if(candidate.evaluation?.acceptance_failures?.length)card.append(el('p','',`Gate: ${candidate.evaluation.acceptance_failures.join(', ').replaceAll('_',' ')}`));
      if(candidate.status==='awaiting_human_approval'){const actions=el('div','candidate-actions');const approve=el('button','button button-green','Approve & promote');approve.addEventListener('click',()=>candidateDecision(candidate.id,'approve'));const reject=el('button','button button-outline','Reject');reject.addEventListener('click',()=>candidateDecision(candidate.id,'reject'));actions.append(approve,reject);card.append(actions);}
      if(['auto_promoted','human_approved'].includes(candidate.status)){const rollback=el('button','button button-outline','Rollback policy');rollback.addEventListener('click',()=>rollbackCandidate(candidate.id));card.append(rollback);}
      candidates.append(card);
    }
}
function pct(value){return value==null?'—':`${(Number(value)*100).toFixed(0)}%`;}
async function candidateDecision(id,decision){try{await api(`/api/candidates/${encodeURIComponent(id)}/decision`,{method:'POST',body:JSON.stringify({decision})});await refreshWorkspace();toast(`Candidate ${decision} recorded.`);}catch(error){toast(error.message);}}
async function rollbackCandidate(id){if(!confirm('Restore the previous policy version?'))return;try{await api(`/api/candidates/${encodeURIComponent(id)}/rollback`,{method:'POST'});await refreshWorkspace();toast('Policy rolled back.');}catch(error){toast(error.message);}}

$('#goal-form').addEventListener('submit',async event=>{
  event.preventDefault();const backend=$('#agent-backend').value;if(!backendReady(backend)){toast(backend==='codex_subscription'?'Sign in to Codex with ChatGPT first (codex login).':'Set OPENAI_KEY for API mode.');return;}if(!state.health?.typesafe_configured){toast('Set TYPESAFE_API_KEY for Jev checks.');return;}
  const request={goal:$('#goal').value,agent_backend:backend,execution_mode:$('#execution-mode').value,data_source:$('#data-source').value,delivery_mode:'sandbox',max_leads:Number($('#max-leads').value),max_tool_calls:Number($('#tool-cap').value),max_model_cost_usd:Number($('#cost-cap').value),audience_exclusions:$('#exclusions').value.split(',').map(x=>x.trim()).filter(Boolean),suppression_list:$('#suppressions').value.split(',').map(x=>x.trim()).filter(Boolean),live_evaboot_authorized:$('#live-auth').checked};
  const button=$('#create-plan');button.disabled=true;button.textContent='Planning…';try{const run=await api('/api/runs',{method:'POST',body:JSON.stringify(request)});state.runId=run.id;renderRun(run);await refreshWorkspace();setView('overview');toast('Bounded plan created. Review it before running.');}catch(error){toast(error.message);}finally{button.disabled=false;button.innerHTML='Create plan <span>→</span>';}
});
$('#execute-run').addEventListener('click',async()=>{if(!state.runId)return;const button=$('#execute-run');button.disabled=true;button.textContent='Agent working…';try{const endpoint=state.run?.status==='deferred'?'resume':'execute';const run=await api(`/api/runs/${encodeURIComponent(state.runId)}/${endpoint}`,{method:'POST'});renderRun(run);await refreshWorkspace();toast(run.status==='awaiting_human_review'?'Run paused for human review.':'Run finished.');}catch(error){toast(error.message);}finally{if(state.run?.status!=='running')button.disabled=false;}});
$('#stop-run').addEventListener('click',async()=>{if(!state.runId)return;try{renderRun(await api(`/api/runs/${encodeURIComponent(state.runId)}/stop`,{method:'POST'}));toast('Stop requested.');}catch(error){toast(error.message);}});
$('#refresh-run').addEventListener('click',()=>state.runId?selectRun(state.runId):refreshWorkspace());
$('#run-improvement').addEventListener('click',async()=>{if(!state.runId)return;const button=$('#run-improvement');button.disabled=true;button.textContent='Evaluating…';try{await api(`/api/runs/${encodeURIComponent(state.runId)}/improve`,{method:'POST'});await refreshWorkspace();setView('learning');toast('Candidate evaluated against the locked test set.');}catch(error){toast(error.message);}finally{button.disabled=false;button.innerHTML='Evaluate policy idea <span>→</span>';}});
$('#data-source').addEventListener('change',renderKeys);
$('#agent-backend').addEventListener('change',()=>{renderKeys();if(state.run)renderRun(state.run);});
$('#new-run').addEventListener('click',()=>{state.runId=null;state.run=null;state.newRun=true;renderRun({id:'',status:'planning',leads:[],events:[],counters:{},plan:null});setView('overview');$('#goal').focus();});
$('#new-run-small').addEventListener('click',()=>$('#new-run').click());
$$('.nav-tab').forEach(button=>button.addEventListener('click',()=>setView(button.dataset.view)));
refreshWorkspace();setInterval(()=>{if(state.run?.status==='running'||state.runId)refreshWorkspace();},7000);
