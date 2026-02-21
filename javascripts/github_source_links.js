/**
 * Adds "View on GitHub" links to mkdocstrings source code blocks.
 *
 * mkdocstrings-python renders source blocks as:
 *   <details class="mkdocstrings-source">
 *     <summary>Source code in <code>{filepath}</code></summary>
 *     <div class="highlight">...line-numbered code...</div>
 *   </details>
 *
 * This script extracts the filepath and first line number from each block
 * and inserts a GitHub link at the top of the expanded content.
 */
document.addEventListener("DOMContentLoaded", function () {
  var repoLink = document.querySelector('a.md-source[data-md-component="source"]');
  if (!repoLink) return;
  var repoBase = repoLink.href.replace(/\/$/, "");

  var sourceBlocks = document.querySelectorAll("details.mkdocstrings-source");

  sourceBlocks.forEach(function (details) {
    var summary = details.querySelector("summary");
    if (!summary) return;

    var codeEl = summary.querySelector("code");
    if (!codeEl) return;

    var filepath = codeEl.textContent.trim();
    if (!filepath) return;

    var lineNo = null;
    var linenosEl = details.querySelector(".linenodiv .normal");
    if (linenosEl) {
      lineNo = parseInt(linenosEl.textContent.trim(), 10);
    }

    var url = repoBase + "/blob/main/" + filepath;
    if (lineNo && !isNaN(lineNo)) {
      url += "#L" + lineNo;
    }

    var linkContainer = document.createElement("div");
    linkContainer.className = "github-source-link";

    var link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="14" height="14" fill="currentColor" style="vertical-align: text-bottom; margin-right: 4px;">' +
      '<path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/>' +
      "</svg>" +
      "View on GitHub";

    linkContainer.appendChild(link);

    var firstChild = summary.nextElementSibling;
    if (firstChild) {
      details.insertBefore(linkContainer, firstChild);
    } else {
      details.appendChild(linkContainer);
    }
  });
});
