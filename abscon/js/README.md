## Bridging Native Mermaid Parser with Python
The repository contains a bundled CommonJS file that exposes a function `parse_mermaid` to parse mermaid. This function can be used bridged to Python with [PythonMonkey](http://pythonmonkey.io/).

## Create Bundle from scratch
```
npm install
npx rollup -c
```

## Know issues
It seems that for now Mermaid does not work with server side node applications [link](https://github.com/mermaid-js/mermaid/issues/5204). Calling Mermaid parsing from server side will result in a DOM error. The solution is to hot patching the sanity check (since hopefully we are not receiving arbitrary test inputs 😄).
* After bundle the code, comment out this piece of code from `navie_parser.bundle` (typically at line 4787):
```js
  if (config2.dompurifyConfig) {
    text = purify.sanitize(sanitizeMore(text, config2), config2.dompurifyConfig).toString();
  } else {
    text = purify.sanitize(sanitizeMore(text, config2), {
      FORBID_TAGS: ["style"]
    }).toString();
  }
```

## Limitation
It seems only works with typical flowcharts (starting with the `graph` keywords). I think other graph types are probably hidden somewhere but I haven't explored them thoroughly. 