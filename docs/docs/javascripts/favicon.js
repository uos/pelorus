// The tab icon follows the browser's colour scheme.
//
// favicon.svg switches its colours with a media query of its own, but browsers
// draw a favicon once and keep it, and Safari ignores the query altogether. So
// this points the icon at favicon-light.svg or favicon-dark.svg instead, and
// again whenever the scheme changes.

(function () {
  "use strict";

  var dark = window.matchMedia("(prefers-color-scheme: dark)");

  function update() {
    var icon = document.querySelector('link[rel="icon"]');
    if (!icon) return;
    var href = icon.href.replace(/favicon(-light|-dark)?\.svg$/, dark.matches ? "favicon-dark.svg" : "favicon-light.svg");
    if (icon.href !== href) icon.href = href;
  }

  update();
  dark.addEventListener("change", update);
  // For browsers that do not report a scheme change to a background tab.
  document.addEventListener("visibilitychange", update);
})();
