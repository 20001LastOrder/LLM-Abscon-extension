import commonjs from '@rollup/plugin-commonjs';
import resolve from '@rollup/plugin-node-resolve';

export default {
  input: 'native_parser.mjs',
  output: {
    file: 'native_parser.bundle.js',
    format: 'cjs',
    inlineDynamicImports: true,
  },
  plugins: [
    resolve({ browser: true, preferBuiltins: false }),
    commonjs(),
  ],
};