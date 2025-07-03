# Documentation for `arrayHelpers.js`

```javascript
function unique(arr) {
  return [...new Set(arr)];
}

function chunk(arr, size) {
  const chunks = [];
  for (let i = 0; i < arr.length; i += size) {
    chunks.push(arr.slice(i, i + size));
  }
  return chunks;
}
```

## Generated Documentation

Produces a unique array of unique values.

@param {Array} arr The array to compact.
@returns {Array} Returns a new array of unique values.
