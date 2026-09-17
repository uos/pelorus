// Copy buttons on the configuration reference pages.
//
// One listener sits on the document, not one on each button. The site uses
// instant navigation, which replaces the page content without a reload and
// would leave per-button listeners behind on discarded nodes.

(function () {
  "use strict";

  var RESET_AFTER_MS = 1600;

  function fallbackCopy(text) {
    // navigator.clipboard needs a secure context, which a local `zensical
    // serve` over plain http is not.
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    var copied = false;
    try {
      copied = document.execCommand("copy");
    } catch (error) {
      copied = false;
    }
    document.body.removeChild(area);
    return copied;
  }

  function feedback(button, state) {
    button.setAttribute("data-state", state);
    window.setTimeout(function () {
      button.removeAttribute("data-state");
    }, RESET_AFTER_MS);
  }

  function copy(button) {
    var line = button.getAttribute("data-line");
    if (!line) {
      return;
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(line).then(
        function () {
          feedback(button, "copied");
        },
        function () {
          feedback(button, fallbackCopy(line) ? "copied" : "failed");
        }
      );
      return;
    }
    feedback(button, fallbackCopy(line) ? "copied" : "failed");
  }

  document.addEventListener("click", function (event) {
    var button = event.target.closest(".rl-copy");
    if (button) {
      copy(button);
    }
  });
})();
