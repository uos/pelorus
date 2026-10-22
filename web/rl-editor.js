// Ratslang syntax highlighting for a textarea, driven by the TextMate grammar
// of the ratslang VS Code extension (copied to syntaxes/ by build.sh).
//
// The textarea keeps its text but draws it transparent. A <pre> behind it shows
// the same text with coloured tokens, so typing, selection, undo and copy and
// paste stay the browser's own.

const GRAMMAR_URL = "syntaxes/ratslang.tmLanguage.json";

/** Annotations in comments, such as @enum and @recommended, which document a setting. */
const ANNOTATION = /@[A-Za-z_][\w-]*/g;

/** The CSS class of a TextMate scope, from the most specific known prefix. */
const SCOPE_CLASSES = [
  ["comment", "rl-comment"],
  ["constant.character.escape", "rl-escape"],
  ["string", "rl-string"],
  ["constant.numeric.quantity", "rl-quantity"],
  ["constant.numeric", "rl-number"],
  ["constant.language", "rl-constant"],
  ["variable.language", "rl-internal"],
  ["entity.name.namespace", "rl-namespace"],
  ["entity.name.type", "rl-value"],
  ["variable", "rl-variable"],
  ["keyword.operator", "rl-operator"],
  ["punctuation", "rl-punctuation"],
];

function scopeClass(scope) {
  return SCOPE_CLASSES.find(([prefix]) => scope.startsWith(prefix))?.[1] ?? "";
}

function sticky(source) {
  return new RegExp(source, "uy");
}

/**
 * The grammar's top-level patterns as an ordered list of rules. A rule is
 * either {match, scope} or {begin, end, scope, inner}.
 */
function compile(grammar) {
  const resolve = (pattern) => {
    if (pattern.include) {
      const entry = grammar.repository[pattern.include.slice(1)];
      return entry ? resolve(entry) : [];
    }
    if (pattern.begin) {
      return [
        {
          begin: sticky(pattern.begin),
          end: sticky(pattern.end),
          scope: pattern.name ?? "",
          inner: (pattern.patterns ?? []).flatMap(resolve),
        },
      ];
    }
    if (pattern.match) return [{ match: sticky(pattern.match), scope: pattern.name ?? "" }];
    return (pattern.patterns ?? []).flatMap(resolve);
  };
  return grammar.patterns.flatMap(resolve);
}

function escapeHtml(text) {
  return text.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
}

function span(scope, text) {
  const cls = scopeClass(scope);
  if (!cls) return escapeHtml(text);
  if (cls === "rl-comment") {
    const marked = escapeHtml(text).replace(ANNOTATION, '<span class="rl-annotation">$&</span>');
    return `<span class="${cls}">${marked}</span>`;
  }
  return `<span class="${cls}">${escapeHtml(text)}</span>`;
}

/** Try `rules` at `position`. Returns [html, next position] or null. */
function matchAt(rules, text, position) {
  for (const rule of rules) {
    if (rule.match) {
      rule.match.lastIndex = position;
      const found = rule.match.exec(text);
      if (found && found[0].length > 0) {
        return [span(rule.scope, found[0]), position + found[0].length];
      }
      continue;
    }
    rule.begin.lastIndex = position;
    const opened = rule.begin.exec(text);
    if (!opened) continue;
    // Inside a begin/end region, until its end or the end of the text.
    let html = escapeHtml(opened[0]);
    let at = position + opened[0].length;
    while (at < text.length) {
      const inner = matchAt(rule.inner, text, at);
      if (inner) {
        html += inner[0];
        at = inner[1];
        continue;
      }
      rule.end.lastIndex = at;
      const closed = rule.end.exec(text);
      if (closed) {
        html += escapeHtml(closed[0]);
        at += closed[0].length;
        break;
      }
      html += escapeHtml(text[at]);
      at++;
    }
    return [`<span class="${scopeClass(rule.scope)}">${html}</span>`, at];
  }
  return null;
}

function highlight(rules, text) {
  let html = "";
  let plain = "";
  let position = 0;
  while (position < text.length) {
    const token = matchAt(rules, text, position);
    if (token) {
      html += escapeHtml(plain) + token[0];
      plain = "";
      position = token[1];
    } else {
      plain += text[position];
      position++;
    }
  }
  // A trailing newline needs content after it, or the <pre> drops the last line.
  return `${html}${escapeHtml(plain)} `;
}

/**
 * Highlight `textarea`. Returns a function to call after setting its value
 * from code, which fires no input event. Without the grammar the textarea
 * stays plain.
 */
export async function ratslangEditor(textarea) {
  let rules;
  try {
    const response = await fetch(GRAMMAR_URL);
    if (!response.ok) return () => {};
    rules = compile(await response.json());
  } catch {
    return () => {};
  }

  const wrapper = document.createElement("div");
  wrapper.className = "rl-editor";
  const view = document.createElement("pre");
  view.setAttribute("aria-hidden", "true");
  textarea.replaceWith(wrapper);
  wrapper.append(view, textarea);

  const refresh = () => {
    view.innerHTML = highlight(rules, textarea.value);
    view.scrollTop = textarea.scrollTop;
    view.scrollLeft = textarea.scrollLeft;
  };
  textarea.addEventListener("input", refresh);
  textarea.addEventListener("scroll", () => {
    view.scrollTop = textarea.scrollTop;
    view.scrollLeft = textarea.scrollLeft;
  });
  refresh();
  return refresh;
}
