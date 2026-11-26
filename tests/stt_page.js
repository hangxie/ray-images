// Runs stt.html's script against fake browser APIs (DOM, Web Audio, fetch): one recording from Record to its text.
// Usage: node stt_page.js <script.js>; exits 1 on a failed check.
const fs = require('fs');
const vm = require('vm');

const elements = {};
const element = id => elements[id] ??= {
  id, textContent: '', className: '', children: [],
  classList: {set: new Set(), add(c) { this.set.add(c); }, remove(c) { this.set.delete(c); }},
  append(...c) { this.children.push(...c); }, scrollIntoView() {},
};
const document = {getElementById: element, createElement: () => element(`_${Object.keys(elements).length}`)};
let port, refuse = true, uploads = [], answer = {status: 200, body: {text: 'Hello world.', language: 'english', duration: 1.0}};
const context = {
  document, console,
  URL: {createObjectURL: () => 'blob:worklet'},
  Blob: class { constructor(parts, options) { this.parts = parts; this.type = options?.type; } },
  FormData: class { constructor() { this.fields = {}; } append(name, value, filename) { this.fields[name] = {value, filename}; } },
  AudioContext: class {
    sampleRate = 48000; audioWorklet = {addModule: async () => {}};
    createMediaStreamSource() { return {connect() {}}; }
    async resume() {}
  },
  AudioWorkletNode: class { constructor() { port = this.port = {}; } },
  navigator: {mediaDevices: {getUserMedia: async () => { if (refuse) { refuse = false; throw new Error('denied'); } return {}; }}},
  fetch: async (url, init) => {
    uploads.push({url, form: init.body});
    return {ok: answer.status === 200, status: answer.status, statusText: 'error', json: async () => answer.body};
  },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);

let failed = 0;
function check(ok, what) {
  console.log(ok ? 'ok  ' : 'FAIL', what);
  if (!ok) failed++;
}
const $ = element;
const settle = () => new Promise(resolve => setTimeout(resolve, 0));

(async () => {
  await $('record').onclick();
  check($('status').textContent === 'Microphone unavailable: denied' && !$('record').classList.set.has('on'),
    'a refused microphone is reported');
  await $('record').onclick();
  check($('record').textContent === 'Stop' && $('record').classList.set.has('on'), 'Record retries and starts recording');
  for (let i = 0; i < 375; i++) port.onmessage({data: new Float32Array(128).fill(0.5)});  // 1 s at 48 kHz
  await $('record').onclick();
  await settle();
  check(uploads.length === 1 && uploads[0].url === 'v1/audio/transcriptions', 'Stop uploads to v1/audio/transcriptions');
  const file = uploads[0].form.fields.file, view = new DataView(file.value.parts[0]);
  check(file.filename === 'speech.wav' && file.value.type === 'audio/wav' && !uploads[0].form.fields.language,
    'as speech.wav, with the language left to the server');
  check(view.getUint32(24, true) === 16000 && view.getUint32(40, true) === 16000 * 2, 'as 1 s of 16 kHz 16-bit WAV');
  check(view.getInt16(44, true) === Math.trunc(0.5 * 0x7fff), 'with the recorded level');
  check($('log').children.at(-1).textContent === 'Hello world.' && /english/.test($('status').textContent),
    'the text is shown');

  await $('record').onclick();
  port.onmessage({data: new Float32Array(128)});
  await $('record').onclick();
  check(uploads.length === 1 && $('status').textContent === 'Too short', 'a too-short recording is not uploaded');

  answer = {status: 503, body: {error: {message: 'busy'}}};
  await $('record').onclick();
  for (let i = 0; i < 375; i++) port.onmessage({data: new Float32Array(128)});
  await $('record').onclick();
  await settle();
  const last = $('log').children.at(-1);
  check(last.className === 'error' && last.textContent === 'Transcription failed: busy', 'an error is shown');
  process.exit(failed ? 1 : 0);
})().catch(err => {
  console.log('ERROR', err);
  process.exit(1);
});
