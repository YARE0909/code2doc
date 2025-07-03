function gcd(a, b) {
  if (b === 0) {
    return a;
  }
  return gcd(b, a % b);
}

function isPalindrome(str) {
  const cleaned = str.replace(/[\W_]/g, '').toLowerCase();
  return cleaned === cleaned.split('').reverse().join('');
}

