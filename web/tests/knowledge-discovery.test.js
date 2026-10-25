import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const appSource = readFileSync(new URL('../src/App.jsx', import.meta.url), 'utf8');
const discoverySource = appSource.slice(
  appSource.indexOf('function VideoAnalysisWorkspace'),
  appSource.indexOf('function UploadCard'),
);

test('video workspace keeps full-video drafting and removes the active Expert Questions workflow', () => {
  assert.doesNotMatch(discoverySource, /Existing SOP proposal records/);
  assert.doesNotMatch(discoverySource, /video-expert-questions|knowledge-discovery\/questions|EXPERT ANSWER|Generate questions/);
  assert.match(appSource, /label: 'Video analysis'/);
  assert.match(discoverySource, /video-sop-drafts/);
  assert.match(discoverySource, /title="Full-video SOP draft"/);
  assert.match(discoverySource, /Generate Full Video SOP Draft/);
});

test('full-video drafting uses the selected source-level draft action without starting analysis', () => {
  const handler = discoverySource.match(/const generateFullVideoSop = async \(\) => \{[\s\S]*?\n  \};/)?.[0] || '';
  assert.match(handler, /onDraft\(Number\(sourceId\)/);
  assert.doesNotMatch(handler, /video-analysis.*method:\s*'POST'/);
  assert.match(handler, /draftingFullSop/);
  assert.doesNotMatch(handler, /running/);
  assert.match(discoverySource, /transcript-only generation/);
  assert.match(discoverySource, /disabled=\{busy \|\| externalBusy \|\| draftingFullSop \|\| !sourceId\}/);
  assert.match(discoverySource, /Visual analysis is displayed above for inspection and never changes the draft/);
});
