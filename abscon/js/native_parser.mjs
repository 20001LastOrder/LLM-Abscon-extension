// // parser.mjs
// import { Window } from "happy-dom";

// function injectDOMPolyfill(context) {
//   const window = new Window();
//   for (const key of Object.getOwnPropertyNames(window)) {
//     if (key in context) continue;
//     try { context[key] = window[key]; } catch { /* ignore */ }
//   }
// }
// injectDOMPolyfill(globalThis);

function replacer(key, value) {
  if(value instanceof Map) {
    return Object.fromEntries(value);
  } else {
    return value;
  }
}

// import mermaid from "mermaid";
if (typeof structuredClone === "undefined") {
  globalThis.structuredClone = (obj) => {
    return JSON.parse(JSON.stringify(obj));
  };
}

import mermaid from "mermaid";
export async function parse_mermaid(text) {
  // const { default: mermaid } = await import("mermaid")
  mermaid.initialize({ startOnLoad: false, securityLevel: "loose" });
  await mermaid.parse(text); // Pre-parse to catch errors early
  const result = (await mermaid.mermaidAPI.getDiagramFromText(text)).db;

  return JSON.stringify(result, replacer);
}

globalThis.parse_mermaid = parse_mermaid;

// Read JSON from stdin -> { "text": "graph TD; A-->B;" } and write result as JSON
// async function main() {
//   // const chunks = [];
//   // for await (const chunk of process.stdin) chunks.push(chunk);
//   // const { text } = JSON.parse(Buffer.concat(chunks).toString("utf8"));
//   const text = "graph TD; A--> |Yes|B[Btest];";
//   const out = await parse_mermaid(text);
//   console.log(out);
// }

// // await main()
// main().catch(err => {
//   console.error(err?.stack || String(err));
//   process.exit(1);
// });
