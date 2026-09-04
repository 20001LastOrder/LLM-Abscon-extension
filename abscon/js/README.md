# Using Mermaid's Parser from Python

This directory contains `native_parser.bundle.js`, a CommonJS bundle that exposes `parse_mermaid`. The Python code loads that function through [PythonMonkey](https://pythonmonkey.io/), allowing AbsCon to use Mermaid's own parser.

## Rebuild the bundle

From this directory, install the JavaScript dependencies and run Rollup:

```bash
npm install
npx rollup -c
```

## Apply the server-side workaround

Mermaid's parser currently expects browser DOM APIs, which causes an error when it runs in a server-side JavaScript environment. The background is tracked in [Mermaid issue #5204](https://github.com/mermaid-js/mermaid/issues/5204).

After rebuilding the bundle, open `native_parser.bundle.js` and remove or comment out the following sanitization block (usually near line 4787):

```js
if (config2.dompurifyConfig) {
  text = purify.sanitize(sanitizeMore(text, config2), config2.dompurifyConfig).toString();
} else {
  text = purify.sanitize(sanitizeMore(text, config2), {
    FORBID_TAGS: ["style"]
  }).toString();
}
```

This patch assumes that the parser only receives trusted input.

## Current limitation

The integration has only been tested with standard flowcharts whose source begins with the `graph` keyword. Other Mermaid diagram types may require more work.
