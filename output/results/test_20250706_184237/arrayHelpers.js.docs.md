# Documentation for `arrayHelpers.js`

```javascript
function chunk(arr, size) {
  const chunks = [];
  for (let i = 0; i < arr.length; i += size) {
    chunks.push(arr.slice(i, i + size));
  }
  return chunks;
}
```

## Generated Documentation

Returns a chunk of the given array @param {Array} arr @param {Number} size @return {Array}
