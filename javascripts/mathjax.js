window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache()
  MathJax.typesetClear()
  MathJax.texReset()

  document.querySelectorAll(".md-nav--secondary .md-ellipsis").forEach(el => {
    if (el.textContent.includes("\\(")) {
      el.classList.add("arithmatex")
    }
  })

  MathJax.typesetPromise()
})