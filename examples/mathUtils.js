function add(a, b) {
  return a + b;
}

function factorial(n) {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}

module.exports = { add, factorial };
