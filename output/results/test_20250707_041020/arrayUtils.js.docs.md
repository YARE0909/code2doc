# Documentation for `arrayUtils.js`

```javascript
function flatten(arr) {
    return arr.reduce((flat, toFlatten) => {
        return flat.concat(Array.isArray(toFlatten) ? flatten(toFlatten) : toFlatten);
    }, []);
}
```

## Generated Documentation

Flatten an array into an array. @param {Array} arr @returns {Array}
