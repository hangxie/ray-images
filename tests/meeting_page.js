// Runs meeting.html's script against fake browser APIs (DOM, Web Audio, WebSocket, storage): the language picker, and
// a meeting from the first audio frame to `done`. Usage: node meeting_page.js <script.js>; exits 1 on a failed check.
const fs = require('fs');
const vm = require('vm');

const elements = {};
function element(id) {
  return elements[id] ??= {
    id, value: '', textContent: '', hidden: false, disabled: false, className: '', style: {}, children: [], options: [],
    classList: {
      set: new Set(),
      toggle(c, on) { on ? this.set.add(c) : this.set.delete(c); }, add(c) { this.set.add(c); }, remove(c) { this.set.delete(c); },
    },
    append(...c) { this.children.push(...c); }, replaceChildren(...c) { this.children = c; },
    scrollIntoView() {}, focus() { document.activeElement = this; }, remove() { this.removed = true; },
    addEventListener(type, fn) { this['on' + type] = fn; },
    get firstChild() { return this.children[0]; },
  };
}
let created = 0;
const document = {
  activeElement: null, visibilityState: 'visible',
  getElementById: element,
  createElement: tag => Object.assign(element(`_${created++}`), {tagName: tag.toUpperCase()}),
  createTextNode: text => ({textContent: text}),
  addEventListener() {},
};
const $ = element;
$('language').tagName = 'SELECT';
$('language').options = ['', 'english', 'chinese', 'spanish', 'french', 'korean', 'japanese', 'other'].map(value => ({value}));

const sockets = [];
class WebSocket {
  static OPEN = 1;
  constructor(url) { this.url = String(url); this.readyState = 0; this.sent = []; sockets.push(this); }
  send(data) { this.sent.push(data); }
  close() { this.readyState = 3; }
}
let port, refuse = true;  // the first request for the microphone is refused, as a remembered refusal is
const storage = {language: 'German'};  // free text saved by the page before the language menu
const context = {
  document, WebSocket, console, setTimeout, setInterval,
  URL: Object.assign(class extends URL {}, {createObjectURL: () => 'blob:worklet', revokeObjectURL() {}}),
  Blob: class {},
  Option: class { constructor(text) { this.text = this.value = text; } },
  AudioContext: class {
    sampleRate = 48000; currentTime = 0; audioWorklet = {addModule: async () => {}};
    createMediaStreamSource() { return {connect() {}}; }
    resume() {}
  },
  AudioWorkletNode: class { constructor() { port = this.port = {}; } },
  location: {href: 'https://apps.example/meeting/', protocol: 'https:'},
  localStorage: {getItem: k => storage[k] ?? null, setItem: (k, v) => { storage[k] = v; }},
  navigator: {mediaDevices: {getUserMedia: async () => { if (refuse) { refuse = false; throw Object.assign(new Error('Permission denied'), {name: 'NotAllowedError'}); } return {}; }}},
  confirm: () => true,
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);

let failed = 0;
function check(ok, what) {
  console.log(ok ? 'ok  ' : 'FAIL', what);
  if (!ok) failed++;
}
const deliver = (socket, event) => socket.onmessage({data: JSON.stringify(event)});

(async () => {
  check($('language').value === 'other' && $('language-other').value === 'german' && !$('language-other').hidden,
    'saved free text restores as Other');
  $('language-other').value = 'klingon';
  $('language-other').oninput();
  check($('language-other').classList.set.has('invalid') && /auto-detect/.test($('status').textContent),
    'an unknown language is marked and falls back to auto-detect');
  $('language').value = 'chinese';
  $('language').onchange();
  check($('language-other').hidden && storage.language === 'chinese', 'a preset hides Other and is saved');

  await $('meeting').onclick();
  check(sockets.length === 0 && /Microphone blocked: allow it/.test($('status').textContent), 'a refused microphone says how to allow it');
  await $('meeting').onclick();
  const socket = sockets[0];
  check(socket?.url === 'wss://apps.example/meeting/api/stream?language=chinese', `stream URL ${socket?.url}`);
  check($('meeting').textContent === 'Stop meeting' && $('meter').classList.set.has('on'), 'the button stops the meeting');
  const frame = new Float32Array(128).fill(0.1);  // what the worklet posts, at 48 kHz
  for (let i = 0; i < 40; i++) port.onmessage({data: frame});
  check(socket.sent.length === 0, 'audio waits for the socket to open');
  socket.readyState = WebSocket.OPEN;
  for (let i = 0; i < 40; i++) port.onmessage({data: frame});
  const pcm = socket.sent.map(b => new Int16Array(b));
  const samples = pcm.reduce((n, p) => n + p.length, 0);
  // 41 frames go once the socket is open, then 38 at the next 0.1 s; the last one waits.
  check(pcm.length === 2 && samples === Math.floor(41 * 128 / 3) + Math.floor(38 * 128 / 3) && pcm[0][0] === Math.trunc(0.1 * 0x7fff),
    `frames go out as 16 kHz PCM (${samples} samples in ${pcm.length} messages)`);
  deliver(socket, {type: 'partial', start: 0, text: '你'});
  deliver(socket, {type: 'partial', start: 0, text: '你好'});
  const live = $('log').children.filter(c => c.className === 'entry pending');
  check(live.length === 1 && live[0].firstChild.textContent === '你好', 'partials update one grey line');
  deliver(socket, {type: 'segment', start: 0, end: 5, text: '你好', language: 'chinese', paragraph: true, seconds: 0.4});
  check(live[0].removed, 'the segment replaces the partial');
  deliver(socket, {type: 'segment', start: 5, end: 9, text: '世界', language: 'chinese', paragraph: false, seconds: 0.3});
  deliver(socket, {type: 'segment', start: 12, end: 14, text: 'Next.', language: 'chinese', paragraph: true, seconds: 0.3});
  const saved = JSON.parse(storage.transcript);
  check(saved.length === 2 && saved[0].text === '你好 世界' && saved[0].segments.length === 2,
    'pieces join into paragraphs');
  // New note in the middle of a meeting: the transcript and the grey line go, and the meeting carries on.
  deliver(socket, {type: 'partial', start: 15, text: 'Half'});
  const half = $('log').children.at(-1);
  $('new-note').onclick();
  check(JSON.parse(storage.transcript).length === 0 && half.removed === true, 'New note clears the transcript');
  deliver(socket, {type: 'partial', start: 15, text: 'Half a'});
  check($('log').children.length === 1 && $('log').children[0].firstChild.textContent === 'Half a',
    'the next partial shows in the new note');
  deliver(socket, {type: 'segment', start: 15, end: 17, text: 'Half a sentence.', language: 'chinese', paragraph: false, seconds: 0.2});
  const fresh = JSON.parse(storage.transcript);
  check(fresh.length === 1 && fresh[0].text === 'Half a sentence.', 'the meeting goes on in the new note');
  $('meeting').onclick();
  check(JSON.parse(socket.sent.at(-1)).type === 'stop' && $('meeting').disabled, 'Stop sends stop and waits');
  deliver(socket, {type: 'done'});
  check(!$('meeting').disabled && $('meeting').textContent === 'Start meeting', 'done ends the meeting');
  port.onmessage({data: frame});  // the microphone keeps running after the meeting
  check(socket.sent.length === pcm.length + 2, 'nothing is sent after the meeting');
  process.exit(failed ? 1 : 0);
})().catch(err => {
  console.log('ERROR', err);
  process.exit(1);
});
