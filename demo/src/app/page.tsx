"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// Same-origin proxy (see app/api/transliterate/route.ts). The browser never
// sees or calls the backend directly; the frontend forwards server-side.
const API = "/api/transliterate";
// Only auto-show suggestions after a clear typing pause, so fast typing isn't
// interrupted with partial-word matches. Press Tab to search immediately.
const TYPE_PAUSE_MS = 650;
const LRU_MAX = 2000;

type Source = "dict" | "cache" | "model" | "offline" | "memory";
type Result = { candidates: string[]; source: Source; latency_ms: number };
type Timing = { source: Source; rtt: number; serverMs: number };
type Suggestion = {
  range: [number, number];
  query: string;
  candidates: string[];
  source: Source;
  appendSpace: boolean;
  rtt: number; // round-trip time the browser saw (ms)
  serverMs: number; // backend compute time reported by the API (ms)
};

// Plain-language labels + explanations for where a suggestion came from.
const SOURCE_LABEL: Record<Source, string> = {
  dict: "dictionary",
  cache: "cached",
  model: "model",
  offline: "offline",
  memory: "memory",
};
const SOURCE_HELP: Record<Source, string> = {
  dict: "Precomputed dictionary: this word was transliterated ahead of time and stored, so it is served from an in-memory table with no model call (sub-millisecond).",
  cache: "Server cache: a recent model result was reused from the server's in-memory cache.",
  model: "Model: computed live by the CTranslate2 INT8 model (used only for words not in the dictionary).",
  offline: "Offline: the backend was unreachable, so this came from the browser's bundled dictionary.",
  memory: "Memory: retrieved from a word you inserted earlier so you can re-pick.",
};

// Session LRU of romanized lookups: instant, works offline once fetched, and is
// what makes prefetched (skipped-past) words show up instantly.
const lru = new Map<string, Result>();
function lruPut(key: string, val: Result) {
  lru.delete(key);
  lru.set(key, val);
  if (lru.size > LRU_MAX) lru.delete(lru.keys().next().value as string);
}

let clientDict: Record<string, string[]> | null = null;
let clientDictLoading: Promise<void> | null = null;
function loadClientDict(): Promise<void> {
  if (clientDict) return Promise.resolve();
  if (!clientDictLoading) {
    clientDictLoading = fetch("/client_dict_hi.json")
      .then((r) => r.json())
      .then((d) => {
        clientDict = d;
      })
      .catch(() => {
        clientDict = {};
      });
  }
  return clientDictLoading;
}

async function lookup(word: string): Promise<Result | null> {
  const key = word.toLowerCase();
  const cached = lru.get(key);
  if (cached) return cached;
  try {
    const res = await fetch(`${API}?word=${encodeURIComponent(key)}&topk=5`);
    if (!res.ok) return offlineLookup(key);
    const body = await res.json();
    const result: Result = { candidates: body.candidates, source: body.source as Source, latency_ms: body.latency_ms };
    lruPut(key, result);
    return result;
  } catch {
    return offlineLookup(key);
  }
}

async function offlineLookup(word: string): Promise<Result | null> {
  await loadClientDict();
  const cands = clientDict?.[word.toLowerCase()];
  if (!cands) return null;
  const r: Result = { candidates: cands, source: "offline", latency_ms: 0 };
  lruPut(word.toLowerCase(), r);
  return r;
}

// Lookup wrapped with the browser-perceived round-trip time (instant for a
// client-cache hit, network+backend for a fresh word).
async function lookupTimed(word: string): Promise<{ result: Result | null; rtt: number }> {
  const t0 = performance.now();
  const result = await lookup(word);
  return { result, rtt: Math.round((performance.now() - t0) * 10) / 10 };
}

const isLatin = (s: string) => /^[A-Za-z]+$/.test(s);

// The whole latin word the cursor sits in or next to.
function wordAtCursor(text: string, pos: number) {
  let s = pos;
  while (s > 0 && /[A-Za-z]/.test(text[s - 1])) s--;
  let e = pos;
  while (e < text.length && /[A-Za-z]/.test(text[e])) e++;
  if (s === e) return null;
  return { word: text.slice(s, e), start: s, end: e };
}

// Pixel position of a character index inside a textarea, via a hidden mirror
// element (textareas don't expose caret coordinates directly).
function caretCoords(el: HTMLTextAreaElement, index: number) {
  const style = getComputedStyle(el);
  const div = document.createElement("div");
  const copy = [
    "boxSizing", "width", "paddingTop", "paddingRight", "paddingBottom",
    "paddingLeft", "borderTopWidth", "borderRightWidth", "borderBottomWidth",
    "borderLeftWidth", "fontFamily", "fontSize", "fontWeight", "lineHeight",
    "letterSpacing", "textTransform", "wordSpacing",
  ] as const;
  copy.forEach((p) => {
    (div.style as unknown as Record<string, string>)[p] =
      (style as unknown as Record<string, string>)[p];
  });
  div.style.position = "absolute";
  div.style.top = "0";
  div.style.left = "0";
  div.style.visibility = "hidden";
  div.style.whiteSpace = "pre-wrap";
  div.style.wordWrap = "break-word";
  div.textContent = el.value.slice(0, index);
  const marker = document.createElement("span");
  marker.textContent = el.value.slice(index) || ".";
  div.appendChild(marker);
  el.parentElement!.appendChild(div);
  const top = marker.offsetTop;
  const left = marker.offsetLeft;
  const height = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.4;
  el.parentElement!.removeChild(div);
  return { top: top - el.scrollTop, left, height };
}

export default function Page() {
  const [text, setText] = useState("");
  const [sugg, setSugg] = useState<Suggestion | null>(null);
  const [pos, setPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const [active, setActive] = useState(0);
  const [lastTiming, setLastTiming] = useState<Timing | null>(null);
  const ta = useRef<HTMLTextAreaElement>(null);
  const typeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prefetchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingCaret = useRef<number | null>(null);

  // Inserted Devanagari word -> its romanization + candidate list, so a wrong
  // pick can be corrected: select the word and choose again.
  const english = useRef<Map<string, { roman: string; candidates: string[] }>>(new Map());

  useEffect(() => {
    if (pendingCaret.current != null && ta.current) {
      ta.current.focus();
      ta.current.setSelectionRange(pendingCaret.current, pendingCaret.current);
      pendingCaret.current = null;
    }
  });

  const place = (el: HTMLTextAreaElement, index: number) => {
    const c = caretCoords(el, index);
    const maxLeft = el.clientWidth - 240;
    setPos({ top: c.top + c.height + 4, left: Math.max(0, Math.min(c.left, maxLeft)) });
  };

  // Background-transliterate every latin word in the text so words you typed
  // past are cached; when the cursor later lands on one, it shows instantly.
  const prefetch = useCallback((value: string) => {
    let matches: string[] = value.match(/[A-Za-z]{2,}/g) || [];
    // Skip the word still being typed (text ends mid-word), so we don't waste
    // lookups on partials like "sunv" / "sunva".
    if (/[A-Za-z]$/.test(value) && matches.length) matches = matches.slice(0, -1);
    const words = Array.from(new Set(matches.map((w) => w.toLowerCase())));
    (async () => {
      for (const w of words) if (!lru.has(w)) await lookup(w);
    })();
  }, []);

  // Show suggestions for the latin word at the caret (instant if prefetched).
  // Reads the live textarea value (not the `text` state, which lags one keystroke
  // behind inside a deferred timer, e.g. searching "rah" while you typed "rahe").
  const showForCaret = useCallback(async () => {
    const el = ta.current;
    if (!el) return;
    if (el.selectionStart !== el.selectionEnd) return; // a range: handled elsewhere
    const value = el.value;
    const w = wordAtCursor(value, el.selectionStart);
    if (!w || w.word.length < 2 || !isLatin(w.word)) {
      setSugg(null);
      return;
    }
    const { result: r, rtt } = await lookupTimed(w.word);
    if (el.value !== value) return; // user kept typing; drop this stale result
    if (!r) return setSugg(null);
    setSugg({
      range: [w.start, w.end], query: w.word, candidates: r.candidates,
      source: r.source, appendSpace: w.end === value.length,
      rtt, serverMs: r.latency_ms,
    });
    setLastTiming({ source: r.source, rtt, serverMs: r.latency_ms });
    setActive(0);
    place(el, w.end);
  }, []);

  const onChange = useCallback((value: string) => {
    setText(value);
    setSugg(null); // hide any stale popover the moment a key is pressed
    if (prefetchTimer.current) clearTimeout(prefetchTimer.current);
    prefetchTimer.current = setTimeout(() => prefetch(value), 500);
    // Only auto-show after the typing pause, so fast typing isn't interrupted.
    if (typeTimer.current) clearTimeout(typeTimer.current);
    typeTimer.current = setTimeout(() => showForCaret(), TYPE_PAUSE_MS);
  }, [prefetch, showForCaret]);

  // Selecting a word re-searches it: latin -> transliterate again; a Devanagari
  // word we inserted -> retrieved from memory so you can re-pick.
  const onSelect = useCallback(() => {
    const el = ta.current;
    if (!el) return;
    const s = el.selectionStart;
    const e = el.selectionEnd;
    if (s >= e) return;
    const selected = el.value.slice(s, e).trim();
    if (!selected) return;
    if (isLatin(selected)) {
      lookupTimed(selected).then(({ result: r, rtt }) => {
        if (!r) return;
        setSugg({ range: [s, e], query: selected, candidates: r.candidates, source: r.source, appendSpace: false, rtt, serverMs: r.latency_ms });
        setLastTiming({ source: r.source, rtt, serverMs: r.latency_ms });
        setActive(0);
        place(el, e);
      });
      return;
    }
    const remembered = english.current.get(selected);
    if (remembered) {
      setSugg({ range: [s, e], query: remembered.roman, candidates: remembered.candidates, source: "memory", appendSpace: false, rtt: 0, serverMs: 0 });
      setLastTiming({ source: "memory", rtt: 0, serverMs: 0 });
      setActive(0);
      place(el, e);
    } else {
      setSugg(null);
    }
  }, []);

  const accept = useCallback((candidate: string) => {
    const el = ta.current;
    if (!sugg || !el) return;
    if (typeTimer.current) clearTimeout(typeTimer.current);
    const value = el.value; // live value, consistent with sugg.range
    const [s, e] = sugg.range;
    const insert = candidate + (sugg.appendSpace ? " " : "");
    english.current.set(candidate, { roman: sugg.query, candidates: sugg.candidates });
    pendingCaret.current = s + insert.length;
    setText(value.slice(0, s) + insert + value.slice(e));
    setSugg(null); // popover disappears on selection
  }, [sugg]);

  const onKeyDown = (ev: React.KeyboardEvent) => {
    // Tab: accept the highlighted match if the popover is open, otherwise search
    // the current word right now (no waiting for the typing pause).
    if (ev.key === "Tab") {
      ev.preventDefault();
      if (typeTimer.current) clearTimeout(typeTimer.current);
      if (sugg) accept(sugg.candidates[active]);
      else showForCaret();
      return;
    }
    if (!sugg) return;
    if (ev.key === "ArrowDown") { ev.preventDefault(); setActive((a) => Math.min(a + 1, sugg.candidates.length - 1)); }
    else if (ev.key === "ArrowUp") { ev.preventDefault(); setActive((a) => Math.max(a - 1, 0)); }
    else if (ev.key === "Enter") { ev.preventDefault(); accept(sugg.candidates[active]); }
    else if (ev.key === "Escape") setSugg(null);
  };

  const saveTxt = () => {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "transliteration.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => () => {
    if (typeTimer.current) clearTimeout(typeTimer.current);
    if (prefetchTimer.current) clearTimeout(prefetchTimer.current);
  }, []);

  return (
    <main className="wrap">
      <div className="head">
        <div>
          <h1>Courtroom notepad</h1>
          <p className="sub">
            Type romanized Hindi; pick from the popover under the word. To fix a
            word, select it and choose again. Save as a .txt file.
          </p>
        </div>
        <div className="tools">
          <button className="btn" onClick={saveTxt} disabled={!text}>Save .txt</button>
          <button className="btn ghost" onClick={() => { setText(""); setSugg(null); }}>Clear</button>
        </div>
      </div>

      <div className="editorWrap">
        <textarea
          ref={ta}
          className="editor"
          value={text}
          placeholder="Start typing... e.g. mera nyayalaya mein"
          onChange={(e) => onChange(e.target.value)}
          onSelect={onSelect}
          onClick={() => showForCaret()}
          onKeyDown={onKeyDown}
          onKeyUp={(e) => {
            // Cursor moved into a word (popover closed): show its cached
            // suggestion instantly. onKeyDown handles nav when the popover is open.
            if (sugg) return;
            if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key))
              showForCaret();
          }}
          onBlur={() => setTimeout(() => setSugg(null), 120)}
          autoFocus
        />
        {sugg && sugg.candidates.length > 0 && (
          <div className="popover" style={{ top: pos.top, left: pos.left }}>
            <div className="qlabel">
              <span>{sugg.query}</span>
              <span className={`badge ${sugg.source}`} title={SOURCE_HELP[sugg.source]}>
                {SOURCE_LABEL[sugg.source]}
              </span>
              <span className="badge time">{sugg.rtt} ms</span>
            </div>
            {sugg.candidates.map((c, i) => (
              <div
                key={c + i}
                className={`row ${i === active ? "active" : ""}`}
                onMouseEnter={() => setActive(i)}
                onMouseDown={(e) => { e.preventDefault(); accept(c); }}
              >
                <span className="rank">{i + 1}</span>
                <span className="cand">{c}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="statusbar">
        {lastTiming ? (
          <>
            <span className="muted">Last lookup</span>
            <span className={`badge ${lastTiming.source}`} title={SOURCE_HELP[lastTiming.source]}>
              {SOURCE_LABEL[lastTiming.source]}
            </span>
            <span className="metric"><b>{lastTiming.rtt} ms</b> round-trip</span>
            {lastTiming.serverMs > 0 && (
              <span className="metric">{lastTiming.serverMs} ms backend compute</span>
            )}
            <span className={`target ${lastTiming.rtt < 100 ? "ok" : "warn"}`}>
              {lastTiming.rtt < 100 ? "within" : "over"} the &lt; 100 ms target
            </span>
          </>
        ) : (
          <span className="muted">Type a word and pause — the suggestion latency shows here.</span>
        )}
      </div>

      <p className="hint">
        The badge on each suggestion shows how it was served:
        <span className="badge dict">dictionary</span> precomputed and served from
        memory (sub-millisecond, no model call);
        <span className="badge model">model</span> computed live for rare words;
        <span className="badge cache">cached</span> a reused recent result;
        <span className="badge offline">offline</span> from the browser when the
        backend is unreachable;
        <span className="badge memory">memory</span> a word you inserted earlier,
        so you can re-pick it.
      </p>
    </main>
  );
}
