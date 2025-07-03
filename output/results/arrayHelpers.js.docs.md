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

module.exports = { unique, chunk };

```

## Generated Documentation

/*  @param String arr  @return String
