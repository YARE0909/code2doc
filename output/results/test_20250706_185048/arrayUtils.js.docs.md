# Documentation for `arrayUtils.js`

```javascript
function flatten(arr) {
    return arr.reduce((flat, toFlatten) => {
        return flat.concat(Array.isArray(toFlatten) ? flatten(toFlatten) : toFlatten);
    }, []);
}
```

## Generated Documentation

Flatten an array of strings. @param {Array} arr @return {Array}
