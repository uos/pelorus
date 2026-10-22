//! Reads `pelorus_core/default.rl` as a documentation source.
//!
//! The default configuration is written for humans first: every option carries a
//! comment above it explaining what it does. This module turns that file into a
//! structure the renderer can walk, keeping the order, the nesting and the
//! comments exactly as they appear in the file.
//!
//! The conventions the file has to follow are checked by [`Parsed::lint`]:
//!
//! * A comment block directly above an option documents it. A comment block
//!   directly above several consecutive options documents all of them.
//! * A comment block followed by a blank line documents the surrounding block.
//! * A line of the form `# A | B | C` lists the accepted values of the option
//!   below it and is rendered separately from the prose.
//! * `# @type`, `# @enum`, `# @min`, `# @max`, exclusive bounds and `# @finite`
//!   use the compiler constraint parser and are shown separately from prose.
//! * `# @recommended` marks an option as part of the starter configuration that
//!   `pelorus config generate recommended` writes.

/// Longest first sentence a top-level block may have, so that the summary
/// column of the reference index stays readable.
const MAX_SUMMARY_CHARS: usize = 80;

/// One configuration option.
#[derive(Debug, Clone)]
pub struct Setting {
    /// Full dotted path, for example `imu.calibration.noise_inflation`.
    pub key: String,
    /// Default as written in the file, or `None` for an option that ships
    /// commented out and has no value until the user sets one.
    pub default: Option<String>,
    /// Value from the commented-out line of an option that has no default, so
    /// that the reference can still show a line worth copying.
    pub example: Option<String>,
    /// Prose from the comment block, one entry per paragraph.
    pub doc: Vec<String>,
    /// Accepted values, when the comment block lists them.
    pub values: Vec<String>,
    /// Whether the option is part of the generated starter configuration.
    pub recommended: bool,
    /// Compiler-owned type, enum, numeric, unit and array rules.
    pub constraints: ratslang::Constraints,
}

/// A `name { ... }` block.
#[derive(Debug, Clone)]
pub struct Section {
    /// Full dotted path, for example `imu.calibration`. Empty for the root.
    pub key: String,
    /// Last path segment, for example `calibration`.
    pub name: String,
    /// Prose describing the block as a whole, one entry per paragraph.
    pub doc: Vec<String>,
    /// Settings and nested blocks in the order they appear in the file.
    pub entries: Vec<Entry>,
}

#[derive(Debug, Clone)]
pub enum Entry {
    Setting(Setting),
    Section(Section),
}

impl Section {
    fn new(key: String, name: String, doc: Vec<String>) -> Self {
        Self {
            key,
            name,
            doc,
            entries: Vec::new(),
        }
    }

    /// Every setting in this block and below it, in file order.
    pub fn settings(&self) -> Vec<&Setting> {
        let mut out = Vec::new();
        self.collect_settings(&mut out);
        out
    }

    fn collect_settings<'a>(&'a self, out: &mut Vec<&'a Setting>) {
        for entry in &self.entries {
            match entry {
                Entry::Setting(setting) => out.push(setting),
                Entry::Section(section) => section.collect_settings(out),
            }
        }
    }
}

/// The parsed default configuration.
#[derive(Debug, Clone)]
pub struct Parsed {
    /// Top-level blocks, in file order.
    pub sections: Vec<Section>,
}

impl Parsed {
    /// Every setting in the file, in file order.
    pub fn settings(&self) -> Vec<&Setting> {
        let mut out = Vec::new();
        for section in &self.sections {
            section.collect_settings(&mut out);
        }
        out
    }

    /// Reports options and blocks that carry no comment, and options whose key
    /// the ratslang compiler does not know about.
    ///
    /// The second half is what keeps the documentation honest: the parser here
    /// reads the file as text, so it could drift from what the compiler
    /// actually assigns. Comparing both key sets makes that drift a CI failure
    /// instead of a wrong reference page.
    pub fn lint(&self, source: &str) -> Vec<String> {
        let mut problems = Vec::new();

        for section in &self.sections {
            if section.doc.is_empty() {
                problems.push(format!("block `{}` has no comment above it", section.key));
            } else {
                // The first sentence becomes this block's row in the reference
                // index, so it has to fit a table cell.
                let summary = crate::render::summary(section);
                let length = summary.chars().count();
                if length > MAX_SUMMARY_CHARS {
                    problems.push(format!(
                        "the first sentence of the comment on block `{}` is {length} characters, \
                         which is too long for the reference index (limit {MAX_SUMMARY_CHARS}). \
                         Lead with a short sentence and put the detail after it.",
                        section.key
                    ));
                }
            }
            lint_section(section, &mut problems);
        }

        let compiled = match ratslang::compile_code(source) {
            Ok(mut evaluated) => {
                evaluated.vars = evaluated.vars.with_drop_warning(false);
                evaluated
            }
            Err(err) => {
                problems.push(format!("default.rl does not compile: {err}"));
                return problems;
            }
        };

        match compiled.vars.constraints() {
            Ok(rules) => {
                for setting in self.settings() {
                    if setting.default.is_some()
                        && setting.constraints
                            != rules.get(&setting.key).cloned().unwrap_or_default()
                    {
                        problems.push(format!(
                            "`{}` documentation constraints differ from the compiler",
                            setting.key
                        ));
                    }
                }
            }
            Err(error) => problems.push(format!("cannot read constraints: {error}")),
        }

        let (mut assigned, _) = compiled.vars.usage_vars();
        assigned.sort();
        let mut parsed: Vec<String> = self.settings().iter().map(|s| s.key.clone()).collect();
        parsed.sort();

        for key in &assigned {
            if !parsed.contains(key) {
                problems.push(format!(
                    "`{key}` is assigned in default.rl but the documentation parser did not find it"
                ));
            }
        }
        for key in &parsed {
            // Options that ship commented out are documented but not assigned.
            let is_unset = self
                .settings()
                .iter()
                .any(|s| &s.key == key && s.default.is_none());
            if !assigned.contains(key) && !is_unset {
                problems.push(format!(
                    "`{key}` is documented but the ratslang compiler does not assign it"
                ));
            }
        }

        problems
    }
}

fn lint_section(section: &Section, problems: &mut Vec<String>) {
    for entry in &section.entries {
        match entry {
            Entry::Setting(setting) => {
                if setting.doc.is_empty() && setting.values.is_empty() {
                    problems.push(format!("`{}` has no comment above it", setting.key));
                }
            }
            Entry::Section(child) => {
                if child.doc.is_empty() && child.entries.is_empty() {
                    problems.push(format!("block `{}` is empty", child.key));
                }
                lint_section(child, problems);
            }
        }
    }
}

/// A comment block being collected, before it is known what it documents.
#[derive(Debug, Clone, Default)]
struct Comment {
    /// Prose paragraphs.
    doc: Vec<String>,
    /// Accepted values found on a `# A | B | C` line.
    values: Vec<String>,
    recommended: bool,
    annotations: Vec<String>,
    /// Text of a trailing `# key = value` line, which marks an option that
    /// ships commented out.
    commented_out: Option<String>,
}

impl Comment {
    fn is_empty(&self) -> bool {
        self.doc.is_empty()
            && self.values.is_empty()
            && self.annotations.is_empty()
            && !self.recommended
            && self.commented_out.is_none()
    }

    fn push_line(&mut self, text: &str) {
        if text == "@recommended" {
            self.recommended = true;
            return;
        }
        if let Some(annotation) = text.strip_prefix('@') {
            self.annotations.push(annotation.to_owned());
            return;
        }
        if text.is_empty() {
            // A bare `#` separates paragraphs.
            self.doc.push(String::new());
            return;
        }
        if let Some(values) = parse_values(text) {
            self.values = values;
            return;
        }
        if is_assignment(text) {
            self.commented_out = Some(text.to_string());
            return;
        }
        match self.doc.last_mut() {
            Some(last) if !last.is_empty() => {
                last.push(' ');
                last.push_str(text);
            }
            Some(last) => *last = text.to_string(),
            None => self.doc.push(text.to_string()),
        }
    }

    fn finish(mut self) -> Self {
        self.doc.retain(|paragraph| !paragraph.is_empty());
        self
    }
}

/// `Enu | Ned | Nwu` on a line of its own lists the accepted values.
fn parse_values(text: &str) -> Option<Vec<String>> {
    if !text.contains('|') {
        return None;
    }
    let values: Vec<String> = text
        .split('|')
        .map(|part| part.trim().to_string())
        .collect();
    if values.len() < 2 {
        return None;
    }
    let looks_like_identifier = |value: &String| {
        !value.is_empty()
            && value
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic())
            && value
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
    };
    values.iter().all(looks_like_identifier).then_some(values)
}

fn is_assignment(text: &str) -> bool {
    split_assignment(text).is_some()
}

/// Splits `key = value` into its two halves, ignoring `==` and lines that do
/// not start with an identifier.
fn split_assignment(text: &str) -> Option<(&str, &str)> {
    let (key, value) = text.split_once('=')?;
    let key = key.trim();
    if key.is_empty() || value.starts_with('=') {
        return None;
    }
    let valid = key
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
        && key.chars().next().is_some_and(|c| c.is_ascii_alphabetic());
    valid.then(|| (key, value.trim()))
}

/// Removes a trailing `# ...` comment from a value, leaving text inside string
/// literals alone.
fn split_trailing_comment(value: &str) -> (&str, Option<&str>) {
    let mut in_string = false;
    for (index, c) in value.char_indices() {
        match c {
            '"' => in_string = !in_string,
            '#' if !in_string => {
                return (value[..index].trim_end(), Some(value[index + 1..].trim()));
            }
            _ => {}
        }
    }
    (value.trim_end(), None)
}

/// Reads the default configuration into [`Parsed`].
pub fn parse(source: &str) -> Result<Parsed, String> {
    let lines: Vec<&str> = source.lines().collect();
    let mut root = Section::new(String::new(), String::new(), Vec::new());
    let mut stack: Vec<Section> = Vec::new();
    // The comment block being collected.
    let mut pending: Option<Comment> = None;
    // The comment block that documents the current run of consecutive options.
    let mut carried: Option<Comment> = None;

    let mut index = 0;
    while index < lines.len() {
        let raw = lines[index];
        let line = raw.trim();
        index += 1;

        if line.is_empty() {
            if let Some(comment) = pending.take() {
                flush_dangling(comment, stack.last_mut().unwrap_or(&mut root))?;
            }
            carried = None;
            continue;
        }

        if let Some(text) = line.strip_prefix('#') {
            pending
                .get_or_insert_with(Comment::default)
                .push_line(text.trim());
            continue;
        }

        if let Some(name) = line.strip_suffix('{') {
            let name = name.trim();
            if name.is_empty() {
                return Err(format!("line {}: block without a name", index));
            }
            let comment = pending.take().unwrap_or_default().finish();
            carried = None;
            let parent_key = stack.last().map_or("", |section| section.key.as_str());
            let key = join_key(parent_key, name);
            stack.push(Section::new(key, name.to_string(), comment.doc));
            continue;
        }

        if line.starts_with('}') {
            if let Some(comment) = pending.take() {
                flush_dangling(comment, stack.last_mut().unwrap_or(&mut root))?;
            }
            carried = None;
            let section = stack
                .pop()
                .ok_or_else(|| format!("line {}: closing brace without a block", index))?;
            let parent = stack.last_mut().unwrap_or(&mut root);
            parent.entries.push(Entry::Section(section));
            continue;
        }

        let Some((name, value_start)) = split_assignment(line) else {
            return Err(format!("line {}: cannot read `{}`", index, line));
        };

        // Matrices and vectors run over several lines.
        let mut value = value_start.to_string();
        while value.matches('[').count() > value.matches(']').count() {
            let Some(next) = lines.get(index) else {
                return Err(format!("line {}: unterminated value", index));
            };
            index += 1;
            value.push(' ');
            value.push_str(next.trim());
        }
        let (value, inline_comment) = split_trailing_comment(&value);

        let comment = match pending.take() {
            Some(comment) => {
                let mut comment = comment.finish();
                // Annotation-only blocks do not break prose shared by adjacent
                // settings. Constraints themselves belong to one assignment.
                if comment.doc.is_empty()
                    && comment.values.is_empty()
                    && !comment.annotations.is_empty()
                {
                    if let Some(previous) = &carried {
                        comment.doc = previous.doc.clone();
                        comment.values = previous.values.clone();
                        comment.recommended |= previous.recommended;
                    }
                }
                let mut shared = comment.clone();
                shared.annotations.clear();
                carried = Some(shared);
                comment
            }
            None => carried.clone().unwrap_or_default(),
        };
        let constraints =
            ratslang::Constraints::from_annotations(comment.annotations.iter().map(String::as_str))
                .map_err(|error| format!("line {index}: {error}"))?;
        let mut doc = comment.doc;
        if let Some(inline) = inline_comment {
            doc.push(inline.to_string());
        }

        let parent_key = stack.last().map_or("", |section| section.key.as_str());
        let setting = Setting {
            key: join_key(parent_key, name),
            default: Some(normalize_value(value)),
            example: None,
            doc,
            values: constraints.allowed.clone().unwrap_or(comment.values),
            constraints,
            recommended: comment.recommended,
        };
        let parent = stack.last_mut().unwrap_or(&mut root);
        parent.entries.push(Entry::Setting(setting));
    }

    if let Some(section) = stack.pop() {
        return Err(format!("block `{}` is never closed", section.key));
    }

    let mut sections = Vec::new();
    for entry in root.entries {
        match entry {
            Entry::Section(section) => sections.push(section),
            Entry::Setting(setting) => {
                return Err(format!(
                    "`{}` sits outside any block, which the renderer has no page for",
                    setting.key
                ));
            }
        }
    }
    Ok(Parsed { sections })
}

/// A comment block that turned out not to document an option: either it
/// describes the surrounding block, or it is an option that ships commented out.
fn flush_dangling(comment: Comment, parent: &mut Section) -> Result<(), String> {
    let comment = comment.finish();
    if comment.is_empty() {
        return Ok(());
    }
    let constraints =
        ratslang::Constraints::from_annotations(comment.annotations.iter().map(String::as_str))
            .map_err(|error| error.to_string())?;
    if let Some(assignment) = &comment.commented_out {
        let (name, value) = split_assignment(assignment).expect("checked when collected");
        parent.entries.push(Entry::Setting(Setting {
            key: join_key(&parent.key, name),
            default: None,
            example: Some(normalize_value(value)),
            doc: comment.doc,
            values: constraints.allowed.clone().unwrap_or(comment.values),
            constraints,
            recommended: comment.recommended,
        }));
        return Ok(());
    }
    parent.doc.extend(comment.doc);
    Ok(())
}

fn join_key(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_string()
    } else {
        format!("{parent}.{name}")
    }
}

/// Collapses the whitespace that keeps matrices readable in the source file.
fn normalize_value(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_was_space = false;
    for c in value.chars() {
        if c.is_whitespace() {
            if !last_was_space {
                out.push(' ');
            }
            last_was_space = true;
        } else {
            out.push(c);
            last_was_space = false;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_comments_values_and_nesting() {
        let parsed = parse(
            r#"
outer {
  # The outer block.

  # @recommended
  # What this does.
  # A | B
  mode = A

  inner {
    # Shared by both.
    first = 1
    second = 2
  }
}
"#,
        )
        .unwrap();

        assert_eq!(parsed.sections.len(), 1);
        let outer = &parsed.sections[0];
        assert_eq!(outer.doc, vec!["The outer block."]);

        let settings = parsed.settings();
        assert_eq!(settings[0].key, "outer.mode");
        assert_eq!(settings[0].default.as_deref(), Some("A"));
        assert_eq!(settings[0].values, vec!["A", "B"]);
        assert!(settings[0].recommended);
        assert_eq!(settings[0].doc, vec!["What this does."]);

        assert_eq!(settings[1].key, "outer.inner.first");
        assert_eq!(settings[1].doc, vec!["Shared by both."]);
        assert_eq!(settings[2].key, "outer.inner.second");
        assert_eq!(settings[2].doc, vec!["Shared by both."]);
    }

    #[test]
    fn reads_multi_line_values_and_commented_out_options() {
        let parsed = parse(
            r#"
outer {
  # A block.

  # The rotation.
  rotation = [ [1.0, 0.0],
               [0.0, 1.0] ]

  # Only used when set.
  # covariance = 0.001

  # Trailing note.
  count = 3 # in samples
}
"#,
        )
        .unwrap();

        let settings = parsed.settings();
        assert_eq!(
            settings[0].default.as_deref(),
            Some("[ [1.0, 0.0], [0.0, 1.0] ]")
        );
        assert_eq!(settings[1].key, "outer.covariance");
        assert_eq!(settings[1].default, None);
        assert_eq!(settings[1].example.as_deref(), Some("0.001"));
        assert_eq!(settings[2].doc, vec!["Trailing note.", "in samples"]);
    }

    #[test]
    fn paragraphs_break_on_a_bare_hash() {
        let parsed = parse(
            r#"
outer {
  # A block.

  # First paragraph
  # continues here.
  #
  # Second paragraph.
  value = 1
}
"#,
        )
        .unwrap();

        assert_eq!(
            parsed.settings()[0].doc,
            vec!["First paragraph continues here.", "Second paragraph."]
        );
    }

    #[test]
    fn the_shipped_default_configuration_parses_and_lints_clean() {
        let source = crate::DEFAULT_CONFIG;
        let parsed = parse(source).expect("default.rl must parse");
        let problems = parsed.lint(source);
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }
}

#[cfg(test)]
mod constraint_tests {
    use super::*;
    #[test]
    fn metadata_preserves_shared_prose_but_does_not_leak_bounds() {
        let source = "outer {\n# A block.\n\n# Shared explanation.\n# @type number\n# @min 1\nfirst = 2\n# @type number\n# @max 10\nsecond = 3\n}\n";
        let parsed = parse(source).unwrap();
        let settings = parsed.settings();
        assert_eq!(settings[0].doc, settings[1].doc);
        assert!(settings[0].constraints.minimum.is_some());
        assert!(settings[1].constraints.minimum.is_none());
        assert!(settings[1].constraints.maximum.is_some());
        assert!(parsed.lint(source).is_empty());
    }
    #[test]
    fn enums_and_commented_out_constraints_use_the_compiler_parser() {
        let source = "outer {\n# A block.\n\n# Choose a backend.\n# @enum CPU, Backend\n# @enum_case insensitive\nbackend = CPU\n\n# Optional count.\n# @type integer\n# @min 1\n# count = 3\n\n}\n";
        let parsed = parse(source).unwrap();
        assert_eq!(parsed.settings()[0].values, ["CPU", "Backend"]);
        assert_eq!(parsed.settings()[1].default, None);
        assert!(parsed.settings()[1].constraints.minimum.is_some());
        assert!(
            parsed.settings()[0]
                .doc
                .iter()
                .all(|p| !p.contains("@enum"))
        );
        assert!(parse("outer {\n# @min nonsense\nvalue = 1\n}\n").is_err());
    }
}
