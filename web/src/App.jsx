import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, ArrowDownToLine, ArrowLeft, ArrowRight, ArrowUpRight, BookOpen, BrainCircuit,
  Check, CheckCircle2, ChevronDown, CircleHelp, ClipboardCheck, Clock3,
  FileText, Gauge, LayoutDashboard, Menu, Play, Plus, Search, ShieldAlert,
  ShieldCheck, Smartphone, Upload, Video, X, XCircle, Wrench
} from 'lucide-react';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const AR_URL = import.meta.env.VITE_AR_URL || 'http://localhost:8080/ar/';
const NAV = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'knowledge', label: 'Knowledge sources', icon: Upload },
  { id: 'ask', label: 'Knowledge assistant', icon: CircleHelp },
  { id: 'review', label: 'SOP review', icon: ClipboardCheck },
  { id: 'discovery', label: 'Video analysis', icon: Video },
  { id: 'training', label: 'Training library', icon: BookOpen },
];
const PAGE = {
  overview: ['Operations overview', 'KNOWLEDGE CONTINUITY'],
  knowledge: ['Knowledge sources', 'CAPTURE & INDEX'],
  ask: ['Knowledge assistant', 'RETRIEVE & VERIFY'],
  review: ['SOP review studio', 'EXPERT QUALITY GATE'],
  discovery: ['Video analysis workspace', 'ANALYZE & SYNTHESIZE'],
  training: ['Published procedures', 'APPROVED TRAINING'],
  trainingdetail: ['Guided training', 'LEARN & PRACTICE'],
};

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const text = await response.text();
  let body;
  try { body = text ? JSON.parse(text) : {}; } catch { body = { detail: text }; }
  if (!response.ok) {
    const error = new Error(body.detail || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return body;
}
const splitLines = (value) => value.split('\n').map((item) => item.trim()).filter(Boolean);
const displayValue = (value) => {
  if (typeof value === 'string' || typeof value === 'number') return String(value);
  if (value && typeof value === 'object') return [value.description, value.instruction, value.text, value.name, value.type].filter(Boolean).join(' — ');
  return '';
};
const procedureEditor = (procedure) => ({
  ...procedure,
  toolsText: (procedure.tools || []).map(displayValue).filter(Boolean).join('\n'),
  warningsText: (procedure.warnings || []).map(displayValue).filter(Boolean).join('\n'),
  steps: (procedure.steps || []).map((step) => ({ ...step })),
});
const clock = (seconds) => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;

export default function App() {
  const [tab, setTab] = useState('overview');
  const [sources, setSources] = useState([]);
  const [procedures, setProcedures] = useState([]);
  const [progress, setProgress] = useState([]);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');
  const [errorKind, setErrorKind] = useState('connectivity');
  const [selected, setSelected] = useState(null);
  const [edit, setEdit] = useState(null);
  const [language, setLanguage] = useState('en');
  const [sourceType, setSourceType] = useState('all');
  const [procedureFilter, setProcedureFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [transcriptDiagnostic, setTranscriptDiagnostic] = useState(null);

  const refresh = async () => {
    setLoading(true);
    try {
      const [sourceRows, procedureRows, progressRows] = await Promise.all([
        request('/api/sources'), request('/api/procedures'), request('/api/training/progress'),
      ]);
      setSources(sourceRows);
      setProcedures(procedureRows);
      setProgress(progressRows);
      setError('');
      setErrorKind('connectivity');
    } catch (err) {
      setError(`Could not load SkillBridge data. ${err.message} Start the API and database, then retry.`);
      setErrorKind('connectivity');
    } finally { setLoading(false); }
  };
  useEffect(() => { refresh(); }, []);

  const act = async (callback) => {
    setBusy(true); setNotice(''); setError('');
    try { await callback(); await refresh(); }
    catch (err) {
      setError(err.message || 'The request could not be completed.');
      setErrorKind(err.status === 422 ? 'evidence' : err.status >= 500 ? 'service' : 'request');
    }
    finally { setBusy(false); }
  };
  const chooseProcedure = (procedure) => {
    setSelected(procedure);
    setEdit(procedureEditor(procedure));
  };
  const createDraft = (sourceId, options = {}) => act(async () => {
    const result = await request(`/api/procedures/draft/${sourceId}?language=${language}`, { method: 'POST' });
    options.onCreated?.(result);
    if (options.open !== false) { chooseProcedure(result); setTab('review'); }
    const limited = (result.warnings || []).some((warning) => displayValue(warning).startsWith('LIMITED DRAFT:'));
    setNotice(result.generation?.reused_existing_draft
      ? `Existing source-level draft version ${result.generation.version} reused. Review it against the latest evidence before approval.`
      : limited ? 'Limited draft created. Review its evidence gaps and every step before approval.' : 'Draft created. Check each step against its source before approval.');
  });
  const upload = (kind, file, transcriptionLanguage) => act(async () => {
    const formData = new FormData();
    formData.append('file', file);
    if (kind === 'video') formData.append('language', transcriptionLanguage);
    const result = await request(kind === 'document' ? '/api/documents/upload' : '/api/videos/upload', {
      method: 'POST', body: formData,
    });
    if (kind === 'video') setTranscriptDiagnostic({ ...result.transcription, segments: result.transcript || [], filename: result.source?.filename });
    setNotice(`${result.message} ${result.chunks_created} searchable chunks created.`);
  });
  const ask = (event) => {
    event?.preventDefault();
    return act(async () => {
      const result = await request('/api/ask', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, source_type: sourceType === 'all' ? null : sourceType, response_language: language }),
      });
      setAnswer(result);
    });
  };
  const saveProcedure = () => act(async () => {
    if (!edit || !selected) return;
    const result = await request(`/api/procedures/${selected.id}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: edit.title, description: edit.description,
        tools: splitLines(edit.toolsText), warnings: splitLines(edit.warningsText),
        steps: edit.steps.map(({ instruction, timestamp, source_ref, evidence_type }) => ({ instruction, timestamp: timestamp || null, source_ref: source_ref || null, evidence_type: evidence_type || 'verification_required' })),
      }),
    });
    chooseProcedure(result); setNotice('Changes saved. The procedure still requires expert approval.');
  });
  const review = (action) => act(async () => {
    const result = await request(`/api/procedures/${selected.id}/review`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }),
    });
    chooseProcedure(result);
    setNotice(action === 'approve' ? 'Procedure approved and published to training.' : 'Procedure rejected. Revise it before requesting another review.');
  });

  const approved = procedures.filter((procedure) => procedure.status === 'approved');
  const drafts = procedures.filter((procedure) => procedure.status === 'draft');
  const filteredProcedures = useMemo(() => procedures.filter((procedure) => {
    const matchesStatus = procedureFilter === 'all' || procedure.status === procedureFilter;
    const matchesSearch = `${procedure.title} ${procedure.description}`.toLowerCase().includes(search.toLowerCase());
    return matchesStatus && matchesSearch;
  }), [procedures, procedureFilter, search]);
  const page = PAGE[tab] || PAGE.overview;

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#overview" onClick={(event) => { event.preventDefault(); setTab('overview'); }} aria-label="SkillBridge AR home">
        <span className="brand-mark"><span/><span/><span/></span>
        <span className="brand-copy"><strong>SkillBridge</strong><small>INDUSTRIAL KNOWLEDGE</small></span>
      </a>
      <div className="side-caption">WORKSPACE</div>
      <nav className="side-nav" aria-label="Main navigation">
        {NAV.map(({ id, label, icon: Icon }) => <button key={id} className={`nav-link ${tab === id || (tab === 'trainingdetail' && id === 'training') ? 'active' : ''}`} onClick={() => setTab(id)}>
          <Icon size={17} strokeWidth={1.8}/><span>{label}</span>{id === 'review' && drafts.length > 0 && <em>{drafts.length}</em>}
        </button>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="review-seal"><ShieldCheck size={16}/></div>
        <div><strong>Expert-led learning</strong><span>Every SOP needs review</span></div>
      </div>
      <div className="side-version">SKILLBRIDGE AR <span>•</span> WORKSPACE</div>
    </aside>

    <main className="workspace">
      <header className="topbar">
        <div className="mobile-brand"><span className="brand-mark"><span/><span/><span/></span><strong>SkillBridge</strong></div>
        <div className="crumb"><span>WORKSPACE</span><b>/</b><span>{page[1]}</span></div>
        <div className="top-actions">
          <label className="language-select"><span>Language</span><select value={language} onChange={(event) => setLanguage(event.target.value)} aria-label="Interface and response language"><option value="en">EN</option><option value="ja">日本語</option></select></label>
          <span className="connection"><i className={error ? 'offline' : ''}/>{error ? 'Offline' : 'Connected'}</span>
          <button className="icon-button" onClick={refresh} aria-label="Refresh data" title="Refresh data"><Activity size={17}/></button>
        </div>
      </header>
      <div className="page-wrap">
        <div className="page-heading">
          <div><div className="overline">{page[1]}</div><h1>{page[0]}</h1></div>
          {tab === 'overview' && <button className="button button-primary" onClick={() => setTab('knowledge')}><Plus size={16}/> Add source</button>}
          {tab === 'trainingdetail' && <button className="button button-quiet" onClick={() => setTab('training')}><ArrowLeft size={16}/> Library</button>}
        </div>

        {error && <div className={`message message-error message-${errorKind}`} role="alert"><XCircle size={17}/><span>{errorKind === 'evidence' && <strong>Evidence check: </strong>}{error}</span><button onClick={refresh}>Retry</button></div>}
        {notice && <div className="message message-success" role="status"><CheckCircle2 size={17}/><span>{notice}</span><button aria-label="Dismiss message" onClick={() => setNotice('')}><X size={16}/></button></div>}

        {tab === 'overview' && <Overview sources={sources} procedures={procedures} approved={approved} progress={progress} loading={loading} onNavigate={setTab} onChoose={chooseProcedure}/>}
        {tab === 'knowledge' && <Knowledge sources={sources} busy={busy} loading={loading} onUpload={upload} onDraft={createDraft} transcript={transcriptDiagnostic}/>}
        {tab === 'ask' && <Assistant question={question} setQuestion={setQuestion} answer={answer} sourceType={sourceType} setSourceType={setSourceType} language={language} setLanguage={setLanguage} busy={busy} sourcesCount={sources.length} onAsk={ask}/>}
        {tab === 'review' && <Review procedures={filteredProcedures} selected={selected} edit={edit} setEdit={setEdit} search={search} setSearch={setSearch} status={procedureFilter} setStatus={setProcedureFilter} busy={busy} onChoose={chooseProcedure} onSave={saveProcedure} onReview={review}/>}
        {tab === 'discovery' && <VideoAnalysisWorkspace sources={sources.filter((source) => source.source_type === 'video')} onDraft={createDraft} externalBusy={busy} onOpenProcedure={(procedure) => { chooseProcedure(procedure); setTab('review'); }}/>}
        {tab === 'training' && <TrainingLibrary procedures={approved} loading={loading} search={search} setSearch={setSearch} onOpen={(procedure) => { chooseProcedure(procedure); setTab('trainingdetail'); }}/>}
        {tab === 'trainingdetail' && selected && selected.status === 'approved' && <TrainingDetail key={selected.id} procedure={selected} busy={busy} onSave={(payload) => act(async () => { await request('/api/training/progress', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }); setNotice('Training progress saved.'); })}/>}
        <footer className="footer"><span>SKILLBRIDGE AR <i>•</i> KNOWLEDGE CONTINUITY</span><span>Training instructions require local expert verification.</span></footer>
      </div>
    </main>
  </div>;
}

function PageIntro({ title, text, action }) { return <div className="section-intro"><div><h2>{title}</h2><p>{text}</p></div>{action}</div>; }
function SectionHeading({ title, detail, action }) { return <div className="section-heading"><div><h2>{title}</h2>{detail && <p>{detail}</p>}</div>{action}</div>; }
function EmptyState({ icon: Icon = FileText, title, text, action }) { return <div className="empty-state"><span className="empty-icon"><Icon size={19}/></span><strong>{title}</strong><p>{text}</p>{action}</div>; }
function Status({ value }) { return <span className={`status status-${value}`}>{value}</span>; }

function Overview({ sources, procedures, approved, progress, loading, onNavigate, onChoose }) {
  const recentProcedures = procedures.slice(0, 5);
  return <>
    <section className="command-banner">
      <div className="command-copy"><span className="banner-label"><span/> KNOWLEDGE OPERATIONS</span><h2>Keep expertise moving.</h2><p>Capture practical know-how, verify it with experienced people, and carry approved guidance onto the floor.</p>
        <div className="banner-actions"><button className="button button-light" onClick={() => onNavigate('knowledge')}><Upload size={15}/> Capture knowledge</button><button className="banner-link" onClick={() => onNavigate('ask')}>Ask a question <ArrowUpRight size={15}/></button></div>
      </div>
      <div className="flow-visual" aria-label="Expert knowledge flows through review into guided training">
        <div className="flow-line"><i/><i/><i/></div>
        <div className="flow-nodes"><div className="flow-node"><span className="flow-icon"><FileText size={17}/></span><b>Capture</b><small>Source knowledge</small></div><div className="flow-node"><span className="flow-icon flow-review"><ShieldCheck size={17}/></span><b>Review</b><small>Expert verified</small></div><div className="flow-node"><span className="flow-icon flow-learn"><Smartphone size={17}/></span><b>Transfer</b><small>Guided practice</small></div></div>
        <div className="flow-foot"><span><i/> Human approval at every release</span><span>01 <b>/</b> 03</span></div>
      </div>
    </section>
    <div className="metric-grid">
      <Metric label="Knowledge sources" value={sources.length} note="Indexed manuals & recordings" icon={FileText}/>
      <Metric label="Procedures" value={procedures.length} note={`${approved.length} approved for learners`} icon={ClipboardCheck}/>
      <Metric label="Recent training records" value={progress.length} note="Latest saved sessions returned" icon={Activity}/>
      <Metric label="Awaiting review" value={procedures.filter((item) => item.status === 'draft').length} note="Drafts need an expert decision" icon={ShieldAlert} warm/>
    </div>
    <div className="overview-grid">
      <section className="surface"><SectionHeading title="Knowledge sources" detail="Recently added to the library" action={<button className="text-action" onClick={() => onNavigate('knowledge')}>View sources <ArrowRight size={14}/></button>}/>
        {loading ? <SkeletonRows/> : sources.length ? <div className="data-list">{sources.slice(0, 5).map((source) => <div className="data-row" key={source.id}><span className="file-type">{source.source_type === 'video' ? <Video size={16}/> : <FileText size={16}/>}</span><div className="row-content"><strong>{source.filename}</strong><small>{source.source_type === 'video' ? 'VIDEO TRANSCRIPT' : 'PDF DOCUMENT'} <i>·</i> SOURCE {source.id}</small></div><span className="indexed-mark"><Check size={13}/> Indexed</span></div>)}</div> : <EmptyState title="No source material yet" text="Upload a PDF or expert recording to begin capturing knowledge." action={<button className="button button-outline" onClick={() => onNavigate('knowledge')}>Add a source <ArrowRight size={14}/></button>}/>}</section>
      <section className="surface"><SectionHeading title="Procedure activity" detail="Latest drafts and approved SOPs" action={<button className="text-action" onClick={() => onNavigate('review')}>Review queue <ArrowRight size={14}/></button>}/>
        {loading ? <SkeletonRows/> : recentProcedures.length ? <div className="data-list">{recentProcedures.map((procedure) => <button className="data-row procedure-row" key={procedure.id} onClick={() => { onChoose(procedure); onNavigate(procedure.status === 'approved' ? 'training' : 'review'); }}><span className="procedure-mark">{procedure.status === 'approved' ? <BookOpen size={16}/> : <ClipboardCheck size={16}/>}</span><span className="row-content"><strong>{procedure.title}</strong><small>{procedure.steps?.length || 0} steps <i>·</i> SOP {procedure.id}</small></span><Status value={procedure.status}/></button>)}</div> : <EmptyState icon={ClipboardCheck} title="No procedures in the workflow" text="Generate a draft from an indexed source, then review it here." action={<button className="button button-outline" onClick={() => onNavigate('knowledge')}>Browse sources <ArrowRight size={14}/></button>}/>}</section>
    </div>
  </>;
}
function Metric({ label, value, note, icon: Icon, warm }) { return <div className="metric"><div className={`metric-symbol ${warm ? 'warm' : ''}`}><Icon size={17}/></div><span className="metric-label">{label}</span><strong>{value}</strong><small>{note}</small></div>; }
function SkeletonRows() { return <div className="skeleton-list" aria-label="Loading records"><i/><i/><i/></div>; }

function Knowledge({ sources, busy, loading, onUpload, onDraft, transcript }) {
  return <>
    <PageIntro title="Bring the source material into one place." text="Index technical documents and expert demonstrations so teams can search, review, and turn them into controlled procedures."/>
    <div className="upload-grid"><UploadCard kind="document" title="PDF manuals" detail="Technical manuals, work instructions, and SOPs" formats="PDF · up to 30 MB" busy={busy} onUpload={onUpload}/><UploadCard kind="video" title="Expert video or audio" detail="Recorded demonstrations and spoken guidance" formats="MP4, MOV, M4V, WEBM, MKV, MP3, WAV, M4A · up to 200 MB" busy={busy} onUpload={onUpload}/></div>
    {transcript && <section className="surface transcript-inspection"><SectionHeading title={`Transcript inspection · ${transcript.filename || 'Video source'}`} detail={`${transcript.segment_count} segments · ${transcript.transcript_characters} characters · ${transcript.detected_language || 'language unknown'}${transcript.language_probability != null ? ` · ${(transcript.language_probability * 100).toFixed(0)}% language confidence` : ''}`}/><p className="transcript-caveat">Language confidence identifies the likely spoken language; it does not verify transcript accuracy or technical terminology.</p>{transcript.warning && <div className="review-callout"><ShieldAlert size={16}/><span>{transcript.warning} Re-upload the source with the correct language or review the transcript before drafting.</span></div>}<div className="transcript-segments">{transcript.segments.map((segment, index) => <p key={index}><span>{segment.source_ref}</span>{segment.text}</p>)}</div></section>}
    <section className="surface source-library"><SectionHeading title="Source library" detail={`${sources.length} indexed ${sources.length === 1 ? 'source' : 'sources'}`}/>
      {loading ? <SkeletonRows/> : sources.length ? <div className="table-wrap"><table className="data-table"><thead><tr><th>Source</th><th>Type</th><th>Identifier</th><th>Added</th><th/></tr></thead><tbody>{sources.map((source) => <tr key={source.id}><td><span className="table-file">{source.source_type === 'video' ? <Video size={16}/> : <FileText size={16}/>}<strong>{source.filename}</strong></span></td><td><span className="type-label">{source.source_type}</span></td><td className="muted-cell">SRC-{String(source.id).padStart(4, '0')}</td><td className="muted-cell">{source.created_at ? new Date(source.created_at).toLocaleDateString() : '—'}</td><td><button className="button button-outline button-small" disabled={busy} onClick={() => onDraft(source.id)}><Plus size={14}/> Draft SOP</button></td></tr>)}</tbody></table></div> : <EmptyState title="Your source library is empty" text="Choose a PDF or recording above. Processing status will be shown after the API confirms indexing."/>}
    </section>
  </>;
}
function VideoAnalysisWorkspace({ sources, onDraft, onOpenProcedure, externalBusy = false }) {
  const [sourceId, setSourceId] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [drafts, setDrafts] = useState([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [draftingFullSop, setDraftingFullSop] = useState(false);
  const [generatedVideoSop, setGeneratedVideoSop] = useState(null);

  const load = async (id) => {
    if (!id) return;
    try {
      const [status, events, savedDrafts] = await Promise.all([
        request(`/api/sources/${id}/video-analysis`),
        request(`/api/sources/${id}/evidence-timeline`),
        request(`/api/sources/${id}/video-sop-drafts`),
      ]);
      setAnalysis(status);
      setTimeline(events);
      setDrafts(savedDrafts.video_sop_drafts || []);
      setError('');
    } catch (err) {
      setError(err.message);
    }
  };
  useEffect(() => { setGeneratedVideoSop(null); setAnalysis(null); setTimeline([]); setDrafts([]); if (sourceId) load(sourceId); }, [sourceId]);
  useEffect(() => {
    if (!sourceId || !['pending','running','extracting_frames','analysing_frames','aligning_evidence','indexing'].includes(analysis?.status)) return;
    const timer = setInterval(() => load(sourceId), 4000);
    return () => clearInterval(timer);
  }, [sourceId, analysis?.status]);
  const act = async (fn) => { setBusy(true); setError(''); try { await fn(); } catch (err) { setError(err.message); } finally { setBusy(false); } };
  const start = () => act(async () => { await request(`/api/sources/${sourceId}/video-analysis`, { method: 'POST' }); await load(sourceId); });
  const running = ['pending','running','extracting_frames','analysing_frames','aligning_evidence','indexing'].includes(analysis?.status);
  const metrics = analysis?.metrics || {};
  const fullDraft = generatedVideoSop || drafts[0] || null;
  const draftManifest = fullDraft?.generation?.evidence_manifest || fullDraft?.evidence_manifest || [];
  const draftWarnings = [...(fullDraft?.procedure?.warnings || fullDraft?.warnings || []), ...(fullDraft?.generation?.coverage_warnings || [])]
    .filter((warning, index, items) => items.findIndex((item) => displayValue(item) === displayValue(warning)) === index);
  const draftEvidenceCounts = fullDraft?.generation?.evidence_counts || {
    transcript_segments: draftManifest.filter((item) => ['transcript_chunk', 'window_transcript'].includes(item.kind)).length,
  };
  const generateFullVideoSop = async () => {
    if (!sourceId || draftingFullSop || externalBusy) return;
    setDraftingFullSop(true);
    setError('');
    try {
      await onDraft(Number(sourceId), { open: false, onCreated: setGeneratedVideoSop });
      await load(sourceId);
    } catch (err) {
      setError(err.message);
    } finally {
      setDraftingFullSop(false);
    }
  };

  return <>
    <PageIntro title="Inspect the recording and draft from its transcript." text="Video analysis remains available for visual inspection. Full-video SOP drafts use only the selected source’s stored timestamped transcript; visual observations, analysis coverage, and expert records do not affect generation."/>
    <section className="surface discovery-toolbar">
      <label className="form-control"><span>VIDEO SOURCE</span><select value={sourceId} onChange={(event) => setSourceId(event.target.value)}><option value="">Select a video source</option>{sources.map((source) => <option key={source.id} value={source.id}>{source.filename} · source {source.id}</option>)}</select></label>
      {sourceId && <><span className="status status-draft">{analysis?.status || 'loading'}</span><button className="button button-primary" disabled={busy || running || analysis?.status === 'completed'} onClick={start}><BrainCircuit size={15}/>{analysis?.status === 'failed' || analysis?.status === 'partially_completed' ? 'Resume analysis' : 'Analyze video'}</button></>}
    </section>
    {error && <div className="message message-error" role="alert"><XCircle size={15}/>{error}</div>}
    {analysis && <div className="analysis-progress"><span>{analysis.status === 'failed' ? `Analysis failed during ${analysis.stage || 'processing'}: ${analysis.error}` : `${analysis.stage || analysis.status} · ${analysis.progress || 0}%`}</span><div><i style={{ width: `${analysis.progress || 0}%` }}/></div><small>{metrics.motion_scan_samples ?? 0} motion samples at {metrics.motion_scan_fps ? Number(metrics.motion_scan_fps).toFixed(2) : '—'} fps · baseline interval {metrics.candidate_baseline_interval_seconds ? `${Number(metrics.candidate_baseline_interval_seconds).toFixed(1)}s` : '—'} · {metrics.candidate_count ?? 0} candidates · {metrics.windows_selected ?? 0} windows · {metrics.detailed_frames_completed ?? 0}/{metrics.detailed_frames_selected ?? 0} detailed frames · {metrics.candidate_windows_screened_out ?? 0} lower-priority windows screened out · {metrics.candidate_windows_not_detailed ?? 0} important or failed windows not detailed · {metrics.vision_requests_completed ?? 0} vision batches</small>{analysis.warnings?.length > 0 && <ul>{analysis.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>}</div>}
    {sourceId && <>
      <section className="surface timeline-panel" id="video-evidence">
        <SectionHeading title="Video analysis timeline" detail="Inspect transcript timing, visual observations, and coverage here. Visual analysis is not used to generate SOP drafts."/>
        {analysis?.contact_sheets?.length > 0 && <div className="contact-sheet-list"><h3>Coarse screening sheets</h3>{analysis.contact_sheets.map((sheet) => <figure key={sheet.number}><img src={`${API}${sheet.url}`} alt={`Candidate frames contact sheet ${sheet.number}`} loading="lazy"/><figcaption>Screening sheet {sheet.number} · candidate labels map to the timestamps below</figcaption></figure>)}</div>}
        {timeline.length ? <div className="evidence-timeline">{timeline.map((entry, index) => entry.kind === 'action_window'
          ? <article className="evidence-entry action-window" key={`window-${entry.window_id}`}><time>{Number(entry.timestamp_start).toFixed(1)}–{Number(entry.timestamp_end).toFixed(1)}s</time><div className="evidence-entry-content"><b>Candidate action window · {entry.status}</b><p>{entry.selection_reason}</p>{entry.transcript_segments?.map((segment, segmentIndex) => <p key={segmentIndex}><b>Transcript</b> {segment.text}</p>)}{entry.summary?.sequence_summary && <p><b>Unverified visual sequence summary</b> {entry.summary.sequence_summary}</p>}{entry.summary?.uncertainties?.length > 0 && <p><b>Uncertainty</b> {entry.summary.uncertainties.join(' · ')}</p>}{entry.comparison?.explanation && <p><b>SOP comparison · {entry.comparison.relationship}</b> {entry.comparison.explanation}</p>}</div></article>
          : <article className="evidence-entry" key={`${entry.frame_id || entry.timestamp_start}-${index}`}><time>{Number(entry.timestamp_start).toFixed(1)}s</time>{entry.frame_url && <img src={`${API}${entry.frame_url}`} alt={`Frame at ${entry.timestamp_start} seconds`} loading="lazy"/>}<div className="evidence-entry-content">{entry.candidate && <small>Candidate score {Number(entry.candidate.score).toFixed(2)} · {entry.candidate.selected_for_detail ? 'selected for detailed analysis' : entry.candidate.is_contact_sample ? 'contact sheet only' : 'coverage sample'} · {entry.candidate.selection_reason || entry.candidate.reasons?.join(', ')}</small>}{entry.frame_status === 'failed' && <p>Visual analysis failed for this frame. Retry will preserve other completed frames.</p>}{entry.audio_text && <div><b>Transcript</b><p>{entry.audio_text}</p></div>}{entry.visual?.description && <div><b>AI visual observation · unverified</b><p>{entry.visual.description}</p></div>}{entry.visual?.uncertainties?.length > 0 && <div><b>Uncertainty</b><p>{entry.visual.uncertainties.join(' · ')}</p></div>}{entry.sequence && <div><b>Sampled sequence · not continuous motion</b><p>{[...(entry.sequence.visible_changes || []), ...(entry.sequence.suggested_but_unproven || []).map((item) => `Unproven: ${item}`)].join(' · ')}</p></div>}</div></article>
        )}</div> : <EmptyState icon={Video} title="No visual timeline yet" text={analysis?.status === 'failed' ? 'Transcript data remains available; resume analysis to continue visual coverage.' : 'Analyze this video to populate timestamped transcript and visual evidence.'}/>}
      </section>
      <section className="surface video-sop-workspace">
        <SectionHeading title="Full-video SOP draft" detail={`Source ${sourceId} · transcript-only generation. ${analysis?.run_id ? `Analysis run ${analysis.run_id} is retained for inspection separately.` : 'Video analysis is optional and does not gate drafting.'}`}/>
        {fullDraft && <article className="expert-question video-sop-result"><div className="question-meta"><span>Full-video draft · version {fullDraft.generation?.version ?? fullDraft.version}</span><Status value={fullDraft.status || fullDraft.procedure?.status || 'draft'}/>{(fullDraft.generation?.run_id ?? fullDraft.run_id) != null && <span>Analysis provenance run {fullDraft.generation?.run_id ?? fullDraft.run_id}</span>}</div><strong>{fullDraft.title || fullDraft.procedure?.title}</strong><p>{fullDraft.description || fullDraft.procedure?.description}</p>{draftWarnings.map((warning, index) => <div className="review-callout" key={index}><ShieldAlert size={15}/><span>{displayValue(warning)}</span></div>)}<small>{Object.entries(draftEvidenceCounts).map(([key, value]) => `${key.replaceAll('_', ' ')}: ${value}`).join(' · ')}</small>{draftManifest.length > 0 && <details className="draft-evidence"><summary>Supporting transcript references ({draftManifest.length})</summary><ul>{draftManifest.map((item, index) => <li key={`${item.kind}-${item.id}-${index}`}><span>{item.kind.replaceAll('_', ' ')}</span><a href="#video-evidence">{item.source_ref}</a>{item.timestamp && <small>{item.timestamp}</small>}</li>)}</ul></details>}<button className="button button-outline button-small" disabled={externalBusy} onClick={() => onOpenProcedure(fullDraft.procedure || fullDraft)}><ClipboardCheck size={14}/> Open in SOP review</button></article>}
        {drafts.slice(1).map((item) => <article className="legacy-proposal" key={`video-sop-${item.version}`}><b>Version {item.version} · {item.procedure.status}</b><span>{item.procedure.title} · run {item.run_id}</span><button className="button button-outline button-small" onClick={() => onOpenProcedure(item.procedure)}>Open version</button></article>)}
        <div className="discovery-sop"><div><b>Generate Full Video SOP Draft</b><p>Uses only this video’s timestamped transcript. Visual analysis is displayed above for inspection and never changes the draft. Drafts remain unapproved; unchanged transcripts reuse their existing version.</p></div><button className="button button-primary" disabled={busy || externalBusy || draftingFullSop || !sourceId} onClick={generateFullVideoSop}>{draftingFullSop || externalBusy ? <><span className="spinner"/> Generating…</> : <><ClipboardCheck size={15}/> Generate Full Video SOP Draft</>}</button></div>
      </section>
    </>}
  </>;
}

function UploadCard({ kind, title, detail, formats, busy, onUpload }) {
  const [file, setFile] = useState(null);
  const [language, setLanguage] = useState('auto');
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const accept = kind === 'document' ? '.pdf,application/pdf' : '.mp4,.mov,.m4v,.webm,.mkv,.mp3,.wav,.m4a,video/*,audio/*';
  const validFile = (chosen) => {
    if (!chosen) return;
    const allowed = kind === 'document' ? chosen.name.toLowerCase().endsWith('.pdf') : /\.(mp4|mov|m4v|webm|mkv|mp3|wav|m4a)$/i.test(chosen.name);
    setFile(allowed ? chosen : null);
  };
  return <section className="upload-panel">
    <div className="upload-heading"><span className={`upload-glyph ${kind}`} >{kind === 'video' ? <Video size={19}/> : <FileText size={19}/>}</span><div><h3>{title}</h3><p>{detail}</p></div></div>
    <button className={`drop-zone ${dragging ? 'dragging' : ''}`} type="button" onClick={() => inputRef.current?.click()} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); validFile(event.dataTransfer.files?.[0]); }}>
      <span className="drop-icon"><ArrowDownToLine size={18}/></span><strong>{file ? file.name : 'Choose a file or drop it here'}</strong><small>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · Ready to upload` : formats}</small>
      <input ref={inputRef} type="file" accept={accept} onChange={(event) => validFile(event.target.files?.[0])} aria-label={`Choose ${kind === 'video' ? 'video or audio' : 'PDF'} file`}/>
    </button>
    {kind === 'video' && <label className="form-control compact-control"><span>TRANSCRIPTION LANGUAGE</span><select value={language} onChange={(event) => setLanguage(event.target.value)}><option value="auto">Auto-detect</option><option value="en">English</option><option value="ja">Japanese · 日本語</option></select></label>}
    <div className="upload-footer"><span><ShieldCheck size={14}/> Processing confirmed by API</span><button className="button button-primary" disabled={!file || busy} onClick={() => onUpload(kind, file, language)}>{busy ? <><span className="spinner"/> Processing…</> : <><Upload size={15}/>{kind === 'video' ? 'Upload & transcribe' : 'Upload & index'}</>}</button></div>
  </section>;
}

function Assistant({ question, setQuestion, answer, sourceType, setSourceType, language, setLanguage, busy, sourcesCount, onAsk }) {
  const [expanded, setExpanded] = useState({});
  return <>
    <PageIntro title="Ask what the indexed material can support." text="Answers are grounded in retrieved passages. Open each reference to inspect the excerpt, page, or video timestamp."/>
    <form className="assistant-composer surface" onSubmit={onAsk}>
      <label className="form-control question-control"><span>YOUR QUESTION</span><textarea rows="4" value={question} onChange={(event) => setQuestion(event.target.value)} maxLength={1000} placeholder="Ask about an inspection, setup sequence, or documented precaution…"/><small>{question.length}/1000</small></label>
      <div className="composer-bottom"><div className="filters"><label className="form-control compact-control"><span>SOURCE TYPE</span><select value={sourceType} onChange={(event) => setSourceType(event.target.value)}><option value="all">All indexed sources</option><option value="document">PDF manuals</option><option value="video">Expert recordings</option></select></label><label className="form-control compact-control"><span>RESPONSE LANGUAGE</span><select value={language} onChange={(event) => setLanguage(event.target.value)}><option value="en">English</option><option value="ja">Japanese · 日本語</option></select></label></div><button className="button button-primary" disabled={busy || question.trim().length < 3 || sourcesCount === 0}>{busy ? <><span className="spinner"/> Searching…</> : <><Search size={16}/> Search knowledge</>}</button></div>
      {sourcesCount === 0 && <p className="inline-hint"><FileText size={14}/> Index a source before searching the knowledge base.</p>}
    </form>
    {answer && <section className="answer-workspace surface"><div className="answer-head"><div><span className="overline">RETRIEVAL RESULT</span><h2>Grounded response</h2></div><span className={`grounded-flag ${answer.grounded ? 'yes' : ''}`}><i/>{answer.grounded ? 'Source material retrieved' : 'No source material found'}</span></div><div className="answer-body">{answer.answer}</div>
      <div className="reference-area"><SectionHeading title={`Source references (${answer.sources?.length || 0})`} detail="Evidence used to prepare this response"/>{answer.sources?.length ? <div className="reference-list">{answer.sources.map((source, index) => <article className="reference-item" key={`${source.chunk_id || index}`}><div className="reference-head"><span className="reference-number">{String(index + 1).padStart(2, '0')}</span><div className="row-content"><strong>{source.filename}</strong><small>{source.source_type?.toUpperCase()} <i>·</i> {source.reference}{source.timestamp_start != null && <i> · {clock(source.timestamp_start)}{source.timestamp_end != null ? `–${clock(source.timestamp_end)}` : ''}</i>}</small></div><button className="icon-button small-icon" aria-expanded={!!expanded[index]} onClick={() => setExpanded((value) => ({ ...value, [index]: !value[index] }))}>{expanded[index] ? <X size={15}/> : <ChevronDown size={15}/>}</button></div>{expanded[index] && <p className="excerpt">{source.excerpt}</p>}</article>)}</div> : <EmptyState title="No supporting passages returned" text="Try another source type or upload material that covers this question."/>}</div>
    </section>}
  </>;
}

function Review({ procedures, selected, edit, setEdit, search, setSearch, status, setStatus, busy, onChoose, onSave, onReview }) {
  const updateStep = (index, key, value) => setEdit((state) => ({ ...state, steps: state.steps.map((step, stepIndex) => stepIndex === index ? { ...step, [key]: value } : step) }));
  const changeReview = (action) => {
    if (!selected) return;
    const prompt = action === 'approve' ? 'Approve this procedure and make it available in the training library?' : 'Reject this procedure? It will not be available to learners.';
    if (window.confirm(prompt)) onReview(action);
  };
  return <>
    <PageIntro title="A human review step protects the training library." text="Inspect draft instructions and their source mapping. Approval publishes the procedure; editing an approved procedure returns it to draft under the existing API rules."/>
    <div className="review-layout"><aside className="surface review-queue"><SectionHeading title="Procedure queue" detail={`${procedures.length} matching records`}/><div className="queue-controls"><label className="search-field"><Search size={15}/><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Find a procedure" aria-label="Search procedures"/></label><select aria-label="Filter by status" value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">All statuses</option><option value="draft">Draft</option><option value="approved">Approved</option><option value="rejected">Rejected</option></select></div>
      <div className="queue-list">{procedures.map((procedure) => <button className={`queue-record ${selected?.id === procedure.id ? 'selected' : ''}`} key={procedure.id} onClick={() => onChoose(procedure)}><span className="queue-record-top"><span>SOP {procedure.id}</span><Status value={procedure.status}/></span><strong>{procedure.title}</strong><small>{procedure.steps?.length || 0} procedure steps</small></button>)}{procedures.length === 0 && <EmptyState icon={ClipboardCheck} title="Nothing in this queue" text="Generate a draft from an indexed source or adjust the filters."/>}</div>
    </aside>
    <section className="surface review-editor">{selected && edit ? <>
      <div className="editor-top"><div><span className="overline">PROCEDURE {selected.id} · {selected.status.toUpperCase()}</span><h2>{selected.status === 'draft' ? 'Review draft' : 'Procedure details'}</h2></div><Status value={selected.status}/></div>
      <div className="review-callout"><ShieldCheck size={17}/><span>Compare each instruction with its cited source. Only approve information verified by a qualified expert.</span></div>
      <label className="form-control"><span>PROCEDURE TITLE</span><input value={edit.title} onChange={(event) => setEdit({ ...edit, title: event.target.value })}/></label>
      <label className="form-control"><span>DESCRIPTION</span><textarea rows="3" value={edit.description} onChange={(event) => setEdit({ ...edit, description: event.target.value })}/></label>
      <div className="editor-columns"><label className="form-control"><span>REQUIRED TOOLS <i>· one per line</i></span><textarea rows="4" value={edit.toolsText} onChange={(event) => setEdit({ ...edit, toolsText: event.target.value })} placeholder="Verify tools against the source"/></label><label className="form-control"><span>SAFETY NOTES <i>· one per line</i></span><textarea rows="4" value={edit.warningsText} onChange={(event) => setEdit({ ...edit, warningsText: event.target.value })} placeholder="Confirm warnings with an expert"/></label></div>
      <div className="steps-editor-heading"><div><h3>Procedure steps</h3><p>Source references stay attached to their step when you edit.</p></div><button className="button button-outline button-small" onClick={() => setEdit((state) => ({ ...state, steps: [...state.steps, { instruction: '', timestamp: null, source_ref: null }] }))}><Plus size={14}/> Add step</button></div>
      <div className="steps-editor">{edit.steps.map((step, index) => <div className="edit-step" key={`${selected.id}-${index}`}><span className="step-index">{String(index + 1).padStart(2, '0')}</span><div className="step-edit-fields"><label className="sr-only" htmlFor={`step-${index}`}>Step {index + 1} instruction</label><textarea id={`step-${index}`} rows="2" value={step.instruction || ''} onChange={(event) => updateStep(index, 'instruction', event.target.value)} placeholder="Enter the reviewed instruction"/><div className="step-source-fields"><label>Source reference<input value={step.source_ref || ''} onChange={(event) => updateStep(index, 'source_ref', event.target.value || null)} placeholder="Manual page or video reference"/></label><label>Timestamp<input value={step.timestamp || ''} onChange={(event) => updateStep(index, 'timestamp', event.target.value || null)} placeholder="e.g. 01:24"/></label></div><label className="evidence-kind">Evidence classification<select value={step.evidence_type || 'verification_required'} onChange={(event) => updateStep(index, 'evidence_type', event.target.value)}><option value="source_supported">Source supported</option><option value="visual_observation">Visual observation</option><option value="expert_confirmed">Expert confirmed</option><option value="approved_sop_reference">Approved SOP reference</option><option value="ai_inferred">AI inferred</option><option value="verification_required">Verification required</option></select></label></div><button className="icon-button remove-step" aria-label={`Remove step ${index + 1}`} onClick={() => setEdit((state) => ({ ...state, steps: state.steps.filter((_, stepIndex) => stepIndex !== index) }))}><X size={15}/></button></div>)}{edit.steps.length === 0 && <p className="no-steps">Add a step after checking the source material.</p>}</div>
      <div className="editor-actions"><button className="button button-outline" onClick={onSave} disabled={busy || !edit.title.trim()}>{busy ? <span className="spinner"/> : <Check size={15}/>} Save changes</button><span className="action-spacer"/>{selected.status !== 'approved' && <button className="button button-danger-outline" onClick={() => changeReview('reject')} disabled={busy}><XCircle size={15}/> Reject</button>}<button className="button button-primary" onClick={() => changeReview('approve')} disabled={busy || selected.status === 'approved' || !edit.steps.length || edit.steps.some((step) => !step.instruction?.trim())}><ShieldCheck size={16}/> Approve & publish</button></div>
    </> : <EmptyState icon={ClipboardCheck} title="Select a procedure to review" text="Drafts, approved procedures, and rejected items are listed here."/>}</section></div>
  </>;
}

function TrainingLibrary({ procedures, loading, search, setSearch, onOpen }) {
  const list = procedures.filter((item) => `${item.title} ${item.description}`.toLowerCase().includes(search.toLowerCase()));
  return <>
    <PageIntro title="Guidance that has passed expert review." text="Only approved procedures are available to learners. Open a procedure to work through its steps, confirm understanding, and launch AR guidance."/>
    <div className="library-toolbar"><label className="search-field"><Search size={16}/><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search approved procedures" aria-label="Search approved procedures"/></label><span><ShieldCheck size={15}/> {procedures.length} approved</span></div>
    {loading ? <div className="surface"><SkeletonRows/></div> : list.length ? <div className="training-list">{list.map((procedure) => <article className="training-record" key={procedure.id}><div className="training-record-main"><div className="training-record-meta"><Status value="approved"/><span>SOP {procedure.id}</span></div><h2>{procedure.title}</h2><p>{procedure.description}</p><div className="procedure-facts"><span><ClipboardCheck size={14}/>{procedure.steps?.length || 0} steps</span><span><Wrench size={14}/>{procedure.tools?.length || 0} tools</span><span><ShieldAlert size={14}/>{procedure.warnings?.length || 0} safety notes</span></div></div><button className="button button-outline" onClick={() => onOpen(procedure)}>Open training <ArrowRight size={15}/></button></article>)}</div> : <section className="surface"><EmptyState icon={BookOpen} title={procedures.length ? 'No matching procedures' : 'No approved procedures yet'} text={procedures.length ? 'Try a different title or description.' : 'Reviewed and approved procedures will appear here for learners.'}/></section>}
  </>;
}

function TrainingDetail({ procedure, busy, onSave }) {
  const [index, setIndex] = useState(0);
  const [checked, setChecked] = useState([]);
  const [quiz, setQuiz] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const steps = procedure.steps || [];
  const current = steps[index];
  const percent = steps.length ? Math.round((checked.length / steps.length) * 100) : 0;
  const arLink = `${AR_URL}?title=${encodeURIComponent(procedure.title)}&step=${encodeURIComponent(current?.instruction || 'Follow the approved procedure.')}&warning=${encodeURIComponent(displayValue(procedure.warnings?.[0]) || 'Follow site safety requirements.')}`;
  return <div className="training-detail"><div className="training-detail-title"><div><span className="status status-approved">APPROVED PROCEDURE</span><h2>{procedure.title}</h2><p>{procedure.description}</p></div><span className="detail-id">SOP {procedure.id}</span></div>
    <div className="training-detail-grid"><section className="surface instruction-panel"><div className="step-overview"><div><span className="overline">CURRENT INSTRUCTION</span><strong>Step {steps.length ? index + 1 : 0} <i>/</i> {steps.length}</strong></div><div className="progress-copy">{percent}% reviewed</div></div><div className="progress-track"><span style={{ width: `${percent}%` }}/></div>
      {current ? <div className="active-instruction"><span className="active-step-num">{String(index + 1).padStart(2, '0')}</span><div><span className="overline">INSTRUCTION · {(current.evidence_type || 'verification_required').replaceAll('_', ' ').toUpperCase()}</span><p>{current.instruction}</p>{(current.timestamp || current.source_ref) && <div className="instruction-source"><FileText size={14}/><span>{current.source_ref || 'Source reference'}{current.timestamp ? ` · ${current.timestamp}` : ''}</span></div>}</div></div> : <EmptyState title="No procedure steps" text="Return to the review studio to check this procedure."/>}
      <label className="training-check"><input type="checkbox" checked={checked.includes(index)} disabled={!current} onChange={(event) => setChecked(event.target.checked ? [...checked, index] : checked.filter((value) => value !== index))}/><span className="fake-check"><Check size={13}/></span>I have reviewed this instruction</label>
      <div className="step-navigation"><button className="button button-outline" disabled={index === 0} onClick={() => setIndex(index - 1)}><ArrowLeft size={15}/> Previous</button><span>{checked.length} of {steps.length} steps reviewed</span><button className="button button-dark" disabled={index >= steps.length - 1} onClick={() => setIndex(index + 1)}>Next step <ArrowRight size={15}/></button></div>
      <a className="ar-launch" href={arLink} target="_blank" rel="noreferrer"><span className="ar-launch-icon"><Smartphone size={19}/></span><span><strong>Open AR guidance</strong><small>View the active step in the configured AR experience</small></span><ArrowUpRight size={17}/></a>
    </section>
    <aside className="training-aside"><section className="surface aside-section"><div className="aside-title"><Wrench size={16}/><h3>Required tools</h3></div>{procedure.tools?.length ? <ul className="plain-list">{procedure.tools.map((tool, item) => <li key={item}>{displayValue(tool)}</li>)}</ul> : <p className="aside-empty">No tools listed.</p>}</section>
      <section className="surface aside-section safety-section"><div className="aside-title"><ShieldAlert size={16}/><h3>Safety notes</h3></div>{procedure.warnings?.length ? <ul className="plain-list">{procedure.warnings.map((warning, item) => <li key={item}>{displayValue(warning)}</li>)}</ul> : <p className="aside-empty">No additional notes listed. Follow local safety requirements.</p>}<small>Stop and ask an expert if conditions differ from this approved procedure.</small></section>
      <section className="surface aside-section quiz-section"><div className="aside-title"><Gauge size={16}/><h3>Knowledge check</h3></div><p>If an instruction is unclear or the observed condition differs, what should you do?</p>{['Guess the next step', 'Stop and ask the trainer', 'Skip the instruction'].map((option) => <label className={`quiz-choice ${quiz === option ? 'selected' : ''}`} key={option}><input type="radio" name="quiz" value={option} checked={quiz === option} onChange={() => { setQuiz(option); setSubmitted(false); }}/>{option}</label>)}{submitted && <div className={`quiz-feedback ${quiz === 'Stop and ask the trainer' ? 'correct' : 'incorrect'}`}>{quiz === 'Stop and ask the trainer' ? 'Correct. Follow the site escalation process.' : 'Review the procedure and consult an expert when uncertain.'}</div>}{submitted ? null : <button className="text-action" disabled={!quiz} onClick={() => setSubmitted(true)}>Check answer <ArrowRight size={14}/></button>}</section>
      <button className="button button-primary save-progress" disabled={busy || !steps.length} onClick={() => onSave({ procedure_id: procedure.id, trainee_name: 'Demo Trainee', completed_steps: checked.length, total_steps: steps.length, quiz_score: quiz === 'Stop and ask the trainer' ? 100 : 0, completed: checked.length === steps.length && steps.length > 0 && quiz === 'Stop and ask the trainer' })}>{busy ? <><span className="spinner"/> Saving…</> : <><CheckCircle2 size={16}/> Save training progress</>}</button>
    </aside></div>
  </div>;
}
