//! Turns the parsed default configuration into the Markdown pages under
//! `docs/docs/configuration/`.

use std::collections::BTreeMap;

use crate::parse::{Entry, Parsed, Section, Setting};

const BANNER: &str = "<!-- Generated from pelorus_core/default.rl by `cargo run -p pelorus_docs`.\n     Edit the comments in that file instead of this page. -->\n";

/// A generated page, keyed by its path below `docs/docs/`.
pub type Pages = BTreeMap<String, String>;

/// The first sentence of a block's description.
///
/// The reference index needs one line per block, and taking it from the comment
/// in `default.rl` is what keeps that index from being a second list to
/// maintain. A new block gets a row by carrying a comment, not by being added
/// here. [`crate::parse::Parsed::lint`] enforces that the sentence stays short
/// enough for a table cell.
pub fn summary(section: &Section) -> String {
    let Some(first) = section.doc.first() else {
        return String::new();
    };
    match first
        .char_indices()
        .find(|(index, c)| *c == '.' && first[index + 1..].starts_with(' '))
    {
        Some((index, _)) => first[..index].to_string(),
        None => first.trim_end_matches('.').to_string(),
    }
}

/// Renders every generated page.
pub fn render(parsed: &Parsed, source: &str) -> Result<Pages, String> {
    let mut pages = Pages::new();

    for section in &parsed.sections {
        pages.insert(
            format!("configuration/reference/{}.md", section.name),
            render_section_page(section),
        );
    }
    pages.insert(
        "configuration/reference/index.md".to_string(),
        render_index_page(parsed),
    );
    pages.insert(
        "configuration/recommended.md".to_string(),
        render_recommended_page(parsed, source)?,
    );

    Ok(pages)
}

fn render_index_page(parsed: &Parsed) -> String {
    let mut out = String::new();
    out.push_str(BANNER);
    out.push_str("\n# Reference\n\n");
    out.push_str(
        "Every option Pelorus accepts, in the order it appears in the default configuration. \
         The text is taken from the comments in `pelorus_core/default.rl`, which is the same \
         file `pelorus config generate full` writes, so these pages and the configuration on \
         your disk stay in step.\n\n",
    );
    out.push_str("| Block | Options | Contents |\n| --- | --- | --- |\n");
    for section in &parsed.sections {
        out.push_str(&format!(
            "| [`{name}`](./{name}.md) | {count} | {summary} |\n",
            name = section.name,
            count = section.settings().len(),
            summary = summary(section),
        ));
    }

    let recommended: Vec<&Setting> = parsed
        .settings()
        .into_iter()
        .filter(|setting| setting.recommended)
        .collect();
    out.push_str(&format!(
        "\nOf the {total} options, {count} are marked as worth reviewing for every new \
         installation. Those are the ones `pelorus config generate recommended` writes into \
         a [starter configuration](../recommended.md).\n",
        total = parsed.settings().len(),
        count = recommended.len(),
    ));

    out
}

/// Deepest heading level a page will emit. HTML stops at `h6`, and a block
/// nested further than that keeps the level of its parent, so the heading
/// level never overflows.
const MAX_HEADING: usize = 6;

/// Heading level for a block whose dotted path has `depth` segments. The
/// top-level block is the page title, so its children start at `h2`.
fn block_heading(depth: usize) -> usize {
    depth.clamp(2, MAX_HEADING)
}

/// Heading level for an option that sits directly in a block of that depth, one
/// below the block itself.
fn setting_heading(depth: usize) -> usize {
    (depth + 1).clamp(2, MAX_HEADING)
}

fn render_section_page(section: &Section) -> String {
    let mut out = String::new();
    out.push_str(BANNER);
    out.push_str(&format!("\n# `{}`\n\n", section.name));
    for paragraph in &section.doc {
        out.push_str(paragraph);
        out.push_str("\n\n");
    }

    out.push_str("| Option | Default | Starter |\n| --- | --- | --- |\n");
    render_summary_rows(section, 1, &mut out);
    out.push('\n');

    render_block(section, 1, &mut out);
    out
}

/// The summary table at the top of a page, rendered as the tree it describes:
/// a row per block, then its options indented below it.
///
/// A key is shown relative to the block it sits in, because the indentation and
/// the block rows already say where that is. The heading further down carries
/// the full dotted key for copying into a configuration file.
fn render_summary_rows(section: &Section, depth: usize, out: &mut String) {
    let indent = "&nbsp;&nbsp;".repeat(depth - 1);

    for entry in &section.entries {
        if let Entry::Setting(setting) = entry {
            out.push_str(&format!(
                "| {indent}[`{name}`](#{anchor}) | {default} | {starter} |\n",
                name = relative_to(&setting.key, section),
                anchor = anchor(&setting.key),
                default = short_default(setting),
                starter = if setting.recommended { "yes" } else { "" },
            ));
        }
    }

    for entry in &section.entries {
        if let Entry::Section(child) = entry {
            out.push_str(&format!(
                "| {indent}[**`{name}`**](#{anchor}) | | |\n",
                name = child.name,
                anchor = anchor(&child.key),
            ));
            render_summary_rows(child, depth + 1, out);
        }
    }
}

/// The part of a key below the block holding it.
///
/// Usually one segment, but ratslang also allows an option to be written as a
/// dotted path, as `qos.reliability` is, and dropping the `qos` there would name
/// something that does not exist.
fn relative_to<'a>(key: &'a str, section: &Section) -> &'a str {
    key.strip_prefix(&format!("{}.", section.key)).unwrap_or(key)
}

/// Renders a block and everything below it.
///
/// Options of a block come before its nested blocks, so that a heading is
/// followed by what belongs to it and not by a deeper heading. Heading
/// level follows the depth of the dotted path, which is what makes the table of
/// contents of a page mirror the structure of the configuration file.
fn render_block(section: &Section, depth: usize, out: &mut String) {
    for entry in &section.entries {
        if let Entry::Setting(setting) = entry {
            render_setting(setting, setting_heading(depth), out);
        }
    }

    for entry in &section.entries {
        let Entry::Section(child) = entry else {
            continue;
        };
        let child_depth = depth + 1;

        // A block that holds nothing but other blocks still gets a heading, so
        // that its children stay under it in the table of contents.
        out.push_str(&format!(
            "{hashes} `{key}` {{ #{anchor} .rl-block }}\n\n",
            hashes = "#".repeat(block_heading(child_depth)),
            key = child.key,
            anchor = anchor(&child.key),
        ));
        for paragraph in &child.doc {
            out.push_str(paragraph);
            out.push_str("\n\n");
        }

        render_block(child, child_depth, out);
    }
}

fn render_setting(setting: &Setting, level: usize, out: &mut String) {
    out.push_str(&format!(
        "{hashes} `{key}` {{ #{anchor} .rl-key }}\n\n",
        hashes = "#".repeat(level),
        key = setting.key,
        anchor = anchor(&setting.key),
    ));

    // The line to paste into a configuration file comes first, so that the
    // heading is immediately followed by the thing most people came for. Even
    // a matrix goes in the chip and not in a code block, so that every option
    // on the page has the same shape and nothing is clipped by a fence that
    // will not wrap.
    if let Some(line) = assignment(setting) {
        out.push_str(&copy_button(&line));
    }

    for paragraph in &setting.doc {
        out.push_str(paragraph);
        out.push_str("\n\n");
    }

    if !setting.values.is_empty() {
        let values: Vec<String> = setting
            .values
            .iter()
            .map(|value| format!("`{value}`"))
            .collect();
        out.push_str(&format!("Accepted values: {}\n\n", values.join(", ")));
    }

    if setting.default.is_none() {
        out.push_str(
            "Not set by default. Copy the line above into your configuration to enable it.\n\n",
        );
    }

    if setting.recommended {
        out.push_str("Part of the [starter configuration](../recommended.md).\n\n");
    }
}

/// The full line for a configuration file, using the default or, for an option
/// that ships commented out, the example value next to it.
fn assignment(setting: &Setting) -> Option<String> {
    let value = setting.default.as_ref().or(setting.example.as_ref())?;
    Some(format!("{} = {}", setting.key, value))
}

/// A button that copies one assignment to the clipboard.
///
/// It shows the line it copies, so it doubles as the statement of the default.
/// The behaviour lives in `docs/javascripts/config-reference.js`, which listens
/// one listener on the document and not one per button, so that it keeps
/// working across instant navigation.
fn copy_button(line: &str) -> String {
    format!(
        "<div class=\"rl-assignment\">\n\
         <button type=\"button\" class=\"rl-copy\" data-line=\"{attr}\" \
         aria-label=\"Copy {attr} to the clipboard\"><code>{text}</code></button>\n\
         </div>\n\n",
        attr = escape_html(line),
        text = escape_html(line),
    )
}

fn escape_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

fn short_default(setting: &Setting) -> String {
    match &setting.default {
        None => "not set".to_string(),
        Some(value) if value.len() > 24 => "see below".to_string(),
        Some(value) => format!("`{value}`"),
    }
}

fn anchor(key: &str) -> String {
    key.replace(['.', '_'], "-")
}

fn render_recommended_page(parsed: &Parsed, source: &str) -> Result<String, String> {
    let mut evaluated = ratslang::compile_code(source)
        .map_err(|err| format!("default.rl does not compile: {err}"))?;
    evaluated.vars = evaluated.vars.with_drop_warning(false);
    let recommended = evaluated
        .to_string_filtered("recommended")
        .ok_or_else(|| "default.rl carries no @recommended annotations".to_string())?;

    let count = parsed
        .settings()
        .iter()
        .filter(|setting| setting.recommended)
        .count();

    let mut out = String::new();
    out.push_str(BANNER);
    out.push_str("\n# Starter configuration\n\n");
    out.push_str(&format!(
        "This is what `pelorus config generate recommended` writes. It contains the {count} \
         options that depend on your hardware or on what you want out of the run, and leaves \
         the remaining {rest} at their defaults. Anything left out of the file keeps the \
         default, so most configuration files stay short.\n\n",
        rest = parsed.settings().len() - count,
    ));
    out.push_str("```sh\npelorus config generate recommended > pelorus.rl\n```\n\n");
    out.push_str("```ratslang title=\"pelorus.rl\"\n");
    out.push_str(recommended.trim_end());
    out.push_str("\n```\n\n");
    out.push_str(
        "Every option above, and the ones it leaves out, are described in the \
         [reference](reference/index.md).\n",
    );
    Ok(out)
}

/// Markers around the generated part of the `nav` table in `zensical.toml`.
const NAV_BEGIN: &str = "# pelorus_docs:begin";
const NAV_END: &str = "# pelorus_docs:end";

/// Fills in the reference entries of the navigation table, turning
/// `zensical.template.toml` into `zensical.toml`.
///
/// zensical needs every page listed explicitly, and a page that is not listed
/// is built but never appears in the sidebar. Generating the list between the
/// markers means adding a block to `default.rl` is the only step needed, and it
/// keeps the nav out of the committed tree along with the pages it points at.
pub fn splice_nav(template: &str, parsed: &Parsed) -> Result<String, String> {
    let lines: Vec<&str> = template.lines().collect();

    let find = |marker: &str| {
        lines
            .iter()
            .position(|line| line.trim() == marker)
            .ok_or_else(|| format!("zensical.template.toml has no `{marker}` line"))
    };
    let begin = find(NAV_BEGIN)?;
    let end = find(NAV_END)?;
    if end <= begin {
        return Err(format!("`{NAV_END}` comes before `{NAV_BEGIN}` in zensical.template.toml"));
    }

    // Keep whatever indentation the surrounding table uses.
    let indent: String = lines[begin]
        .chars()
        .take_while(|c| c.is_whitespace())
        .collect();

    let mut generated = vec![format!("{indent}\"configuration/reference/index.md\",")];
    for section in &parsed.sections {
        generated.push(format!(
            "{indent}{{ \"{name}\" = \"configuration/reference/{name}.md\" }},",
            name = section.name,
        ));
    }

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    out.extend(lines[..=begin].iter().map(|line| line.to_string()));
    out.extend(generated);
    out.extend(lines[end..].iter().map(|line| line.to_string()));

    let mut text = out.join("\n");
    if template.ends_with('\n') {
        text.push('\n');
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    const NAV: &str = concat!(
        "nav = [\n",
        "  { \"Reference\" = [\n",
        "      # pelorus_docs:begin\n",
        "      \"configuration/reference/index.md\",\n",
        "      { \"stale\" = \"configuration/reference/stale.md\" },\n",
        "      # pelorus_docs:end\n",
        "  ] },\n",
        "]\n",
    );

    fn two_blocks() -> crate::parse::Parsed {
        parse(
            "# First block. With a second sentence that stays off the index.\n\
             alpha {\n  # An option.\n  a = 1\n}\n\n\
             # Second block.\n\
             beta {\n  # Another option.\n  b = 2\n}\n",
        )
        .unwrap()
    }

    #[test]
    fn the_index_summary_is_the_first_sentence_of_the_block_comment() {
        let parsed = two_blocks();
        assert_eq!(summary(&parsed.sections[0]), "First block");
        assert_eq!(summary(&parsed.sections[1]), "Second block");
    }

    #[test]
    fn splicing_the_nav_replaces_every_reference_entry() {
        let spliced = splice_nav(NAV, &two_blocks()).unwrap();
        assert!(spliced.contains("{ \"alpha\" = \"configuration/reference/alpha.md\" },"));
        assert!(spliced.contains("{ \"beta\" = \"configuration/reference/beta.md\" },"));
        // The block that no longer exists is gone, and the indentation and the
        // lines outside the markers are untouched.
        assert!(!spliced.contains("stale"));
        assert!(spliced.starts_with("nav = [\n  { \"Reference\" = [\n      # pelorus_docs:begin"));
        assert!(spliced.ends_with("]\n"));

        // Running it again changes nothing, which is what lets --check compare.
        assert_eq!(splice_nav(&spliced, &two_blocks()).unwrap(), spliced);
    }

    #[test]
    fn every_option_gets_exactly_one_line_to_copy() {
        let parsed = parse(crate::DEFAULT_CONFIG).unwrap();
        let pages = render(&parsed, crate::DEFAULT_CONFIG).unwrap();

        let buttons: usize = pages
            .values()
            .map(|page| page.matches("class=\"rl-copy\"").count())
            .sum();
        assert_eq!(buttons, parsed.settings().len());

        // Options that ship commented out still offer their example value.
        let unset = parsed
            .settings()
            .into_iter()
            .find(|setting| setting.default.is_none())
            .expect("default.rl has at least one commented-out option");
        assert_eq!(
            assignment(unset).as_deref(),
            Some("wheel.covariance = 0.001")
        );
    }

    #[test]
    fn a_value_with_quotes_is_escaped_for_the_button_attribute() {
        let button = copy_button("lidar.topic = \"/ouster/points\"");
        assert!(button.contains("data-line=\"lidar.topic = &quot;/ouster/points&quot;\""));
        assert!(!button.contains("data-line=\"lidar.topic = \"/"));
    }

    #[test]
    fn splicing_the_nav_fails_without_markers() {
        assert!(splice_nav("nav = []\n", &two_blocks()).is_err());
    }
}
