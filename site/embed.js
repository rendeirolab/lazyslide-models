/* LazySlide model zoo — a self-mounting browser for lazyslide-models.
 *
 * Loads the catalogue published beside this script and renders it into
 * #lazyslide-models. Used by both the standalone page and the LazySlide
 * docs, so it must survive being dropped into a host page: everything is
 * scoped to .lsm-root and it never touches globals it did not create.
 *
 * Hallmark · genre: modern-minimal · macrostructure: Catalogue
 */
(function () {
  "use strict";

  // currentScript is only readable synchronously.
  var SELF = document.currentScript;
  var BASE = new URL(".", SELF ? SELF.src : location.href);
  var MOUNT_ID = "lazyslide-models";
  var HOME = "https://rendeiro.group/lazyslide-models/";
  var REPO = "https://github.com/rendeirolab/lazyslide-models";
  var DOCS = "https://lazyslide.readthedocs.io";

  /* ── html helpers ─────────────────────────────────────────────── */

  var ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (c) {
      return ESCAPES[c];
    });
  }

  function raw(markup) {
    return { __html: markup };
  }

  function html(strings) {
    var values = Array.prototype.slice.call(arguments, 1);
    return strings.reduce(function (out, chunk, i) {
      if (i === 0) return chunk;
      var value = values[i - 1];
      if (value == null || value === false) value = "";
      else if (value && value.__html !== undefined) value = value.__html;
      else if (Array.isArray(value)) value = value.join("");
      else value = esc(value);
      return out + value + chunk;
    }, "");
  }

  /* ── icons ────────────────────────────────────────────────────────
     One family, drawn here: 16px box, 1.4 stroke, round joins. Mixing in
     a second icon set — or standing an emoji in for a glyph — is the tell
     this set exists to avoid.                                         */

  var GLYPH = {
    unlock:
      '<path d="M4.6 7V5.1a3.4 3.4 0 0 1 6.5-1.4"/>' +
      '<rect x="3" y="7" width="10" height="6.6" rx="1.6"/>',
    cross: '<path d="M4.2 4.2 11.8 11.8M11.8 4.2 4.2 11.8"/>',
    lock:
      '<path d="M4.6 7V5.1a3.4 3.4 0 0 1 6.8 0V7"/>' +
      '<rect x="3" y="7" width="10" height="6.6" rx="1.6"/>',
    copy:
      '<rect x="5.8" y="5.8" width="7.7" height="7.7" rx="1.6"/>' +
      '<path d="M10.6 3.9A1.6 1.6 0 0 0 9 2.4H4.1A1.6 1.6 0 0 0 2.5 4v4.9a1.6 1.6 0 0 0 1.5 1.6"/>',
    check: '<path d="M3.2 8.6 6.5 11.9 12.8 4.6"/>',
    paper:
      '<path d="M4 2.2h5.1l2.9 2.9v8.7H4z"/><path d="M9.1 2.2v2.9H12"/>' +
      '<path d="M6.1 8.6h3.8M6.1 11h2.8"/>'
  };

  // Brand marks, not UI icons: solid single paths on a 24 grid, straight
  // from the projects themselves. They are identifiers, so they are drawn as
  // published rather than redrawn to match the stroke family above.
  var BRAND = {
    github: 'M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12',
    huggingface: 'M1.4446 11.5059c0 1.1021.1673 2.1585.4847 3.1563-.0378-.0028-.0691-.0058-.1058-.0058-.4209 0-.8015.16-1.0704.4512-.3454.3737-.4984.8335-.4316 1.293a1.576 1.576 0 0 0 .2148.5978c-.2319.1864-.4018.4456-.4844.7578-.0646.2448-.131.7543.2149 1.2794a1.4552 1.4552 0 0 0-.0625.1055c-.208.3923-.2207.8372-.0371 1.25.2783.6258.9696 1.1175 2.3126 1.6467.8356.3292 1.5988.5411 1.6056.543 1.1046.2847 2.104.4277 2.969.4277 1.4173 0 2.4754-.3849 3.1525-1.1446 1.538.2651 2.791.1403 3.592.006.6773.7555 1.7332 1.1387 3.1467 1.1387.8649 0 1.8643-.143 2.969-.4278.0068-.0019.77-.2138 1.6056-.543 1.343-.5292 2.0343-1.0208 2.3126-1.6466.1836-.4129.171-.8577-.037-1.25a1.4685 1.4685 0 0 0-.0626-.1056c.346-.525.2795-1.0346.2149-1.2793-.0826-.3122-.2525-.5714-.4844-.7579.11-.1816.1831-.3788.2148-.5977.0669-.4595-.0862-.9193-.4316-1.293-.2688-.2913-.6495-.4513-1.0704-.4513-.0209 0-.0376.0008-.0588.0018.3162-.9966.4846-2.0518.4846-3.1523 0-5.807-4.7362-10.5144-10.5789-10.5144-5.8426 0-10.5788 4.7073-10.5788 10.5144Zm10.5788-9.4831c5.2727 0 9.5476 4.246 9.5476 9.483a9.4201 9.4201 0 0 1-.2696 2.2365c-.0039-.0047-.0079-.011-.0117-.0156-.274-.3255-.6679-.5059-1.1075-.5059-.352 0-.714.1155-1.0763.3438-.2403.1517-.5058.422-.7793.7598-.2534-.3492-.608-.5832-1.0137-.6465a1.5174 1.5174 0 0 0-.2344-.0176c-.9263 0-1.4828.7993-1.6935 1.5177-.1046.2426-.6065 1.3482-1.3614 2.0978-1.1681 1.1601-1.4458 2.3534-.8396 3.6382-.843.1029-1.5836.0927-2.365-.006.5906-1.212.3626-2.4388-.8426-3.6322-.755-.7496-1.2568-1.8552-1.3614-2.0978-.2107-.7184-.7673-1.5177-1.6935-1.5177-.078 0-.1568.0054-.2344.0176-.4057.0633-.7604.2973-1.0137.6465-.2735-.3379-.539-.6081-.7794-.7598-.3622-.2283-.7243-.3438-1.0762-.3438-.4266 0-.8094.171-1.0821.4786a9.4208 9.4208 0 0 1-.2598-2.1936c0-5.237 4.2749-9.483 9.5475-9.483zM8.6443 7.0036c-.4838.0043-.9503.2667-1.1934.7227-.3536.6633-.1006 1.4873.5645 1.84.351.1862.4883-.5261.836-.6485.3107-.1095.841.399 1.0078.086.3536-.6634.1025-1.4874-.5625-1.84a1.3659 1.3659 0 0 0-.6524-.1602Zm6.8403 0c-.2199-.002-.4426.05-.6504.1602-.665.3526-.9181 1.1766-.5645 1.84.1669.313.6971-.1955 1.0079-.086.3476.1224.4867.8347.838.6485.6649-.3527.916-1.1767.5624-1.84-.243-.456-.7096-.7184-1.1934-.7227Zm-9.7565 1.418a.8768.8768 0 0 0-.877.877c0 .4846.3925.877.877.877a.8768.8768 0 0 0 .877-.877.8768.8768 0 0 0-.877-.877zm12.6434 0c-.4845 0-.879.3925-.879.877 0 .4846.3945.877.879.877a.8768.8768 0 0 0 .877-.877.8768.8768 0 0 0-.877-.877zM8.7927 11.459c-.179-.003-.2793.1107-.2793.416 0 .8097.3874 2.125 1.4279 2.924.207-.7123 1.3453-1.2832 1.5079-1.2012.2315.1167.2191.4417.6074.7266.3884-.285.374-.6098.6056-.7266.1627-.082 1.3009.4889 1.5079 1.2012 1.0404-.799 1.4278-2.1144 1.4278-2.924 0-1.2212-1.583.6402-3.5413.6485-1.4686-.0061-2.7266-1.0558-3.2639-1.0645zM4.312 14.4768c.5792.365 1.6964 2.2751 2.1056 3.0177.1371.2488.371.3536.582.3536.4188 0 .7465-.4138.0391-.9395-1.0636-.791-.6914-2.0846-.1836-2.1642a.4302.4302 0 0 1 .0664-.004c.4616 0 .666.7892.666.7892s.5959 1.4898 1.6213 2.508c.942.9356 1.062 1.703.4961 2.6661-.0164-.004-.0159.0236-.1484.2149-.1853.2673-.4322.4688-.7188.6152-.5062.2269-1.1397.2696-1.7833.2696-1.037 0-2.1017-.1824-2.6975-.336-.0293-.0075-3.6505-.9567-3.1916-1.8224.0771-.1454.2033-.2031.3633-.2031.6463 0 1.823.9551 2.3283.9551.113 0 .196-.0865.2285-.2031.2249-.8045-3.2787-1.0522-2.9846-2.1642.0519-.1967.193-.2757.3907-.2754.854 0 2.7704 1.4923 3.172 1.4923.0307 0 .0525-.0085.0645-.0274.2012-.3227.1096-.5865-1.3087-1.4395-1.4182-.8533-2.4315-1.329-1.8653-1.9416.0651-.0707.1574-.1015.2695-.1015.8611.0002 2.8948 1.84 2.8948 1.84s.5487.5683.8809.5683c.0762 0 .1416-.0315.1855-.1054.2355-.3946-2.1858-2.2183-2.3224-2.971-.0926-.51.0641-.7676.3555-.7676-.0006.008.1701-.0285.4942.1759zm16.2257.5918c-.1366.7526-2.5579 2.5764-2.3224 2.9709.044.074.1092.1055.1855.1055.3321 0 .881-.5684.881-.5684s2.0336-1.8397 2.8947-1.84c.1121 0 .2044.0308.2695.1016.5662.6125-.447 1.0882-1.8653 1.9415-1.4183.853-1.51 1.1168-1.3087 1.4396.012.0188.0337.0273.0644.0273.4016 0 2.3181-1.4923 3.1721-1.4923.1977-.0002.3388.0787.3907.2754.294 1.112-3.2095 1.3597-2.9846 2.1642.0325.1166.1156.2032.2285.2032.5054 0 1.682-.9552 2.3283-.9552.16 0 .2862.0577.3633.2032.459.8656-3.1623 1.8149-3.1916 1.8224-.5958.1535-1.6605.336-2.6975.336-.6351 0-1.261-.0409-1.7638-.2599-.2949-.1472-.5488-.3516-.7383-.625-.0411-.0682-.1026-.1476-.1426-.205-.5726-.9679-.455-1.7371.4903-2.676 1.0254-1.0182 1.6212-2.508 1.6212-2.508s.2044-.7891.666-.7891a.4318.4318 0 0 1 .0665.0039c.5078.0796.88 1.3732-.1836 2.1642-.7074.5257-.3797.9395.039.9395.211 0 .445-.1047.5821-.3535.4092-.7426 1.5264-2.6527 2.1056-3.0178.5588-.3524.99-.1816.8497.5918z'
  };

  function icon(name) {
    return (
      '<svg class="lsm-ico" viewBox="0 0 16 16" aria-hidden="true" focusable="false">' +
      GLYPH[name] +
      "</svg>"
    );
  }

  function brand(name) {
    return (
      '<svg class="lsm-ico lsm-brand" viewBox="0 0 24 24" aria-hidden="true" ' +
      'focusable="false"><path d="' + BRAND[name] + '"/></svg>'
    );
  }

  /* ── vocabulary ───────────────────────────────────────────────── */

  // The categories and the example snippets are both computed in
  // scripts/build_site.py, next to the registry that defines them. Nothing
  // here decides what a model is or how it is called.

  var USE_LABEL = {
    commercial: "Commercial use allowed",
    noncommercial: "Commercial use not allowed"
  };

  /* ── state ────────────────────────────────────────────────────── */

  var state = { q: "", cats: [], gating: "", use: "", license: "" };

  var data = null;
  var root = null;
  var panel = null;
  var lastFocused = null;
  var writeHash = false;
  var embedded = false;

  function readHash() {
    var hash = location.hash.replace(/^#/, "");
    if (hash.indexOf("lsm:") !== 0) return null;
    var params = new URLSearchParams(hash.slice(4));
    state.q = params.get("q") || "";
    state.cats = (params.get("type") || "").split(",").filter(Boolean);
    state.gating = params.get("gating") || "";
    state.use = params.get("use") || "";
    state.license = params.get("license") || "";
    return true;
  }

  function syncHash() {
    // Only ever own a hash we already own. A Sphinx page has its own
    // anchors, and replacing #install with #lsm:task=vision would break the
    // link the reader arrived on. Standalone, the hash is ours to write.
    if (!writeHash) return;
    var current = location.hash.replace(/^#/, "");
    if (embedded && current && current.indexOf("lsm") !== 0) return;
    var params = new URLSearchParams();
    if (state.q) params.set("q", state.q);
    if (state.cats.length) params.set("type", state.cats.join(","));
    if (state.gating) params.set("gating", state.gating);
    if (state.use) params.set("use", state.use);
    if (state.license) params.set("license", state.license);
    var query = params.toString();
    var next = query ? "#lsm:" + query : location.pathname + location.search;
    history.replaceState(null, "", next);
  }

  /* ── search + filter ──────────────────────────────────────────── */

  function haystack(model) {
    return [
      model.keys.join(" "),
      model.name,
      model.description || "",
      model.categories.join(" "),
      model.tasks.join(" "),
      model.license || "",
      model.vision_encoder || "",
      model.citation ? model.citation.title || "" : ""
    ]
      .join(" ")
      .toLowerCase();
  }

  function score(model, tokens) {
    if (!tokens.length) return 1;
    var best = 0;
    for (var i = 0; i < tokens.length; i++) {
      var token = tokens[i];
      if (model._hay.indexOf(token) === -1) return 0; // every token must hit
      var here = 10;
      if (model.keys.indexOf(token) !== -1) here = 100;
      else if (model.keys.some(function (k) { return k.indexOf(token) === 0; })) here = 80;
      else if (model._name.indexOf(token) === 0) here = 70;
      else if (model.keys.some(function (k) { return k.indexOf(token) !== -1; })) here = 60;
      else if (model._name.indexOf(token) !== -1) here = 50;
      else if ((model.description || "").toLowerCase().indexOf(token) !== -1) here = 20;
      if (here > best) best = here;
    }
    return best;
  }

  function matchesFacets(model, skip) {
    if (skip !== "cat" && state.cats.length) {
      var hit = state.cats.some(function (c) { return model.categories.indexOf(c) !== -1; });
      if (!hit) return false;
    }
    if (skip !== "gating" && state.gating) {
      if (state.gating === "gated" && !model.is_gated) return false;
      if (state.gating === "open" && model.is_gated) return false;
    }
    if (skip !== "use" && state.use) {
      if (state.use === "commercial" && model.commercial !== true) return false;
      if (state.use === "noncommercial" && model.commercial !== false) return false;
    }
    if (skip !== "license" && state.license && model.license_family !== state.license) {
      return false;
    }
    return true;
  }

  function selection(skip) {
    var tokens = state.q.toLowerCase().split(/\s+/).filter(Boolean);
    return data.models
      .map(function (model) {
        return { model: model, score: score(model, tokens) };
      })
      .filter(function (row) {
        return row.score > 0 && matchesFacets(row.model, skip);
      })
      .sort(function (a, b) {
        if (b.score !== a.score) return b.score - a.score;
        return a.model.keys[0].localeCompare(b.model.keys[0]);
      })
      .map(function (row) { return row.model; });
  }

  function countFor(skip, predicate) {
    return selection(skip).filter(predicate).length;
  }

  /* ── views ────────────────────────────────────────────────────── */

  function chip(label, pressed, action, value, count, disabled, extra) {
    return html`<button type="button" class="lsm-chip${raw(extra ? " " + extra : "")}"
      aria-pressed="${pressed ? "true" : "false"}"
      data-action="${action}" data-value="${value}"${raw(disabled ? " disabled" : "")}
      >${label}${raw(count == null ? "" : html`<small>${count}</small>`)}</button>`;
  }

  /** The registry key, styled as the string it is and copyable in one click. */
  function keyButton(key, large) {
    return html`<button type="button" class="lsm-keycopy${raw(large ? " is-lg" : "")}"
      data-action="copykey" data-key="${key}" title="Copy ${key}"
      aria-label="Copy the registry key ${key}"
      ><span class="lsm-key">${key}</span
      ><span class="lsm-ico-slot">${raw(icon("copy"))}</span></button>`;
  }

  /** The access filter wears exactly what the rows it selects wear. */
  function statusChip(value, label, glyph, variant, count) {
    var pressed = state.gating === value;
    return html`<button type="button" class="lsm-chip is-status ${raw(variant)}"
      aria-pressed="${pressed ? "true" : "false"}" data-action="gating" data-value="${value}"
      >${raw(icon(glyph))}${label}<small>${count}</small></button>`;
  }

  /** Commercial use reads faster as a mark than as the word "not allowed". */
  function useChip(value, glyph, count) {
    var pressed = state.use === value;
    return html`<button type="button" class="lsm-chip is-icon is-${raw(glyph)}"
      aria-pressed="${pressed ? "true" : "false"}" data-action="use" data-value="${value}"
      aria-label="${USE_LABEL[value]}" title="${USE_LABEL[value]}"
      >${raw(icon(glyph === "ok" ? "check" : "cross"))}<small>${count}</small></button>`;
  }

  function tag(label, extra, title) {
    return html`<span class="lsm-tag${raw(extra ? " " + extra : "")}"
      ${raw(title ? html`title="${title}"` : "")}>${label}</span>`;
  }

  /** Access is a two-sided fact, so both sides carry a tag. */
  function statusTag(model) {
    return model.is_gated
      ? html`<span class="lsm-tag is-no"
          title="Request access on Hugging Face, then log in with a token"
          >${raw(icon("lock"))}Gated</span>`
      : html`<span class="lsm-tag is-ok" title="No access request needed"
          >${raw(icon("unlock"))}Open</span>`;
  }

  /** A check or a cross, meaningless on its own — always rendered against a
      label: the licence it qualifies, the facet heading, or the table row. */
  function useMark(model) {
    if (model.commercial == null) return "";
    var key = model.commercial ? "commercial" : "noncommercial";
    return html`<span class="lsm-mark ${raw(model.commercial ? "is-ok" : "is-no")}"
      role="img" aria-label="${USE_LABEL[key]}" title="${USE_LABEL[key]}"
      >${raw(icon(model.commercial ? "check" : "cross"))}</span>`;
  }

  function ledeView() {
    var date = (data.generated_at || "").slice(0, 10);
    return html`
      <p class="lsm-headline">A collection of foundation models
        for pathology applications</p>
      <div class="lsm-lede">
        <p class="lsm-count"><b>${data.stats.n_models}</b> models</p>
        <p class="lsm-issue lsm-mono">v${data.version} · indexed ${date}</p>
      </div>`;
  }

  function statsView() {
    var counts = data.stats.by_category;
    var entries = Object.keys(counts).map(function (category) {
      var pressed = state.cats.indexOf(category) !== -1;
      var empty = counts[category] === 0;
      // A declared category with nothing in it yet stays visible and inert —
      // it says "planned", which a missing row could not.
      return html`<button type="button" class="lsm-stat" data-action="cat"
        data-value="${category}" aria-pressed="${pressed ? "true" : "false"}"
        ${raw(empty ? "disabled" : "")}
        ><b>${counts[category]}</b><i>${category}</i></button>`;
    });
    return html`<div class="lsm-stats">${raw(entries.join(""))}</div>`;
  }

  function controlsView(shown) {
    var licenses = Object.keys(data.stats.by_license);
    var active = state.q || state.cats.length || state.gating || state.use || state.license;
    return html`
      <div class="lsm-controls">
        <div class="lsm-searchrow">
          <label class="lsm-search">
            <span class="lsm-sr">Search models</span>
            <input type="search" id="lsm-q" placeholder="Search a name, key, licence or paper"
              value="${state.q}" autocomplete="off" spellcheck="false">
            <span class="lsm-search-key" aria-hidden="true">/</span>
          </label>
          <p class="lsm-showing" role="status">showing ${shown} of ${data.stats.n_models}</p>
          ${raw(active ? '<button type="button" class="lsm-reset" data-action="reset">Clear filters</button>' : "")}
        </div>
        <div class="lsm-facets">
          <div class="lsm-facet">
            <span class="lsm-facet-label">access</span>
            ${raw(statusChip("open", "Open", "unlock", "is-ok",
              countFor("gating", function (m) { return !m.is_gated; })))}
            ${raw(statusChip("gated", "Gated", "lock", "is-no",
              countFor("gating", function (m) { return m.is_gated; })))}
          </div>
          <div class="lsm-facet">
            <span class="lsm-facet-label">commercial use</span>
            ${raw(useChip("commercial", "ok",
              countFor("use", function (m) { return m.commercial === true; })))}
            ${raw(useChip("noncommercial", "no",
              countFor("use", function (m) { return m.commercial === false; })))}
          </div>
          <div class="lsm-facet">
            <span class="lsm-facet-label">licence</span>
            ${raw(licenses.map(function (name) {
              var n = countFor("license", function (m) { return m.license_family === name; });
              var pressed = state.license === name;
              // An empty facet is disabled, but never the active one — that
              // would leave the reader unable to undo their own filter.
              return chip(name, pressed, "license", name, n, n === 0 && !pressed);
            }).join(""))}
          </div>
        </div>
      </div>`;
  }

  /** One model, one line. Columns align because every row shares a template. */
  function rowView(model) {
    var size = [];
    if (model.param_size) size.push(model.param_size);
    if (model.encode_dim) size.push(model.encode_dim + "-d");

    return html`
      <article class="lsm-row" id="lsm-${model.keys[0]}">
        <div class="lsm-cell lsm-cell-model">
          <h2 class="lsm-name" title="${model.name}"><button type="button" class="lsm-open"
            data-action="open" data-key="${model.keys[0]}">${model.name}</button></h2>
          <span class="lsm-keys">${raw(model.keys.map(function (k) {
            return keyButton(k, false);
          }).join(""))}</span>
        </div>
        <p class="lsm-cell lsm-desc" title="${model.description}">${model.description}</p>
        <p class="lsm-cell lsm-cell-type" title="${model.categories.join(", ")}"
          >${model.categories.join(" · ")}</p>
        <p class="lsm-cell lsm-cell-size lsm-mono">${size.join(" · ")}</p>
        <p class="lsm-cell lsm-cell-licence lsm-mono" title="${model.license}"
          >${model.license}${raw(useMark(model))}</p>
        <div class="lsm-cell lsm-cell-access">${raw(statusTag(model))}</div>
      </article>`;
  }

  function listView(models) {
    if (!models.length) {
      return html`<p class="lsm-empty">No model matches that. <button type="button"
        class="lsm-reset" data-action="reset">Clear the filters</button></p>`;
    }
    return html`
      <div class="lsm-rows">
        <div class="lsm-row lsm-head" aria-hidden="true">
          <span class="lsm-cell">Model</span>
          <span class="lsm-cell">Description</span>
          <span class="lsm-cell">Type</span>
          <span class="lsm-cell">Size</span>
          <span class="lsm-cell">Licence</span>
          <span class="lsm-cell">Access</span>
        </div>
        ${raw(models.map(rowView).join(""))}
      </div>`;
  }

  function colophonView() {
    return html`
      <footer class="lsm-colophon lsm-mono">
        <p>v${data.version_full || data.version} · ${data.stats.n_models} models ·
          <a href="${REPO}">source</a> · <a href="${DOCS}">docs</a></p>
      </footer>`;
  }

  /* ── detail panel ─────────────────────────────────────────────── */

  function highlight(code) {
    return code
      .split("\n")
      .map(function (line) {
        return line.indexOf("#") === 0
          ? html`<span class="lsm-cmt">${line}</span>`
          : esc(line);
      })
      .join("\n");
  }

  function specRows(model) {
    var rows = [];
    function row(label, value) {
      if (value == null || value === "") return;
      rows.push(html`<tr><th scope="row">${label}</th><td>${value}</td></tr>`);
    }
    // Access is already stated by the tag beside the key, and the API task
    // rides along so `list_models("vision")` stays discoverable.
    row("type", raw(model.categories.map(function (c, i) {
      return html`${c} <span class="lsm-apitask">${model.tasks[i % model.tasks.length]}</span>`;
    }).join(", ")));
    row("parameters", model.param_size);
    row("embedding dim", model.encode_dim);
    row("FLOPs", model.flops);
    row("vision encoder", model.vision_encoder);
    if (model.input_constraint) {
      var c = model.input_constraint;
      var parts = [];
      if (c.min) parts.push("min " + c.min + " px");
      if (c.max) parts.push("max " + c.max + " px");
      if (c.divisible_by) parts.push("divisible by " + c.divisible_by);
      row("input size", parts.join(" · "));
    }
    if (model.classes) row("output classes", model.classes.join(", "));
    row("licence", model.license_url
      ? raw(html`<a href="${model.license_url}" target="_blank" rel="noopener">${model.license}</a>`)
      : model.license);
    if (model.commercial != null) {
      row("commercial use", raw(useMark(model)));
    }
    return rows.join("");
  }

  function paramsView(model) {
    if (!model.params.length) return "";
    var rows = model.params.map(function (p) {
      var meta = [];
      if (p.annotation) meta.push(html`<code>${p.annotation}</code>`);
      meta.push(p.required
        ? html`<span class="lsm-req">required</span>`
        : html`default <code>${p.default}</code>`);
      return html`<tr><th scope="row"><code>${p.name}</code></th>
        <td>${raw(meta.join(" · "))}</td></tr>`;
    });
    return html`
      <p class="lsm-sub">Constructor arguments</p>
      <table class="lsm-table lsm-params">${raw(rows.join(""))}</table>`;
  }

  function citationView(model) {
    var c = model.citation;
    if (!c) return "";
    var meta = [c.venue, c.year].filter(Boolean).join(" · ");
    return html`
      <div class="lsm-cite">
        <p><span class="lsm-cite-title">${c.title}</span></p>
        <p>${c.author} · ${meta}${raw(c.doi ? html` · <a href="https://doi.org/${c.doi}"
          target="_blank" rel="noopener">doi:${c.doi}</a>` : "")}</p>
      </div>`;
  }

  function openPanel(model) {
    var links = [];
    function link(url, mark, label) {
      links.push(html`<a href="${url}" target="_blank" rel="noopener"
        >${raw(mark)}${label}</a>`);
    }
    if (model.hf_url) link(model.hf_url, brand("huggingface"), "Hugging Face");
    if (model.github_url) link(model.github_url, brand("github"), "GitHub");
    if (model.paper_url) link(model.paper_url, icon("paper"), "Paper");
    if (model.citation) {
      links.push(html`<button type="button" class="lsm-copy-bib"
        data-key="${model.keys[0]}">${raw(icon("copy"))}Copy BibTeX</button>`);
    }

    panel.innerHTML = html`
      <div class="lsm-panel-body">
        <div class="lsm-panel-head">
          <div>
            <h2 class="lsm-name">${model.name}</h2>
            ${raw(model.description ? html`<p class="lsm-desc">${model.description}</p>` : "")}
          </div>
          <button type="button" class="lsm-close" data-action="close" aria-label="Close">
            <svg viewBox="0 0 16 16" class="lsm-ico" aria-hidden="true" focusable="false"
              ><path d="M4 4l8 8M12 4l-8 8"/></svg>
          </button>
        </div>
        <div class="lsm-keybar">
          ${raw(model.keys.map(function (k) { return keyButton(k, true); }).join(""))}
          ${raw(statusTag(model))}
        </div>
        <table class="lsm-table">${raw(specRows(model))}</table>
        ${raw(paramsView(model))}
        ${raw(model.example ? html`
          <p class="lsm-sub">Example</p>
          <div class="lsm-snippet">
            <pre><code>${raw(highlight(model.example))}</code></pre>
            <button type="button" class="lsm-copy" data-action="copy">Copy</button>
          </div>` : "")}
        ${raw(citationView(model))}
        <div class="lsm-links">${raw(links.join(""))}</div>
      </div>`;

    panel._model = model;
    lastFocused = document.activeElement;
    panel.showModal();
  }

  /* ── wiring ───────────────────────────────────────────────────── */

  function render() {
    var models = selection(null);
    var mount = root.querySelector(".lsm-shell");
    var focused = document.activeElement;
    var caret = focused && focused.id === "lsm-q" ? focused.selectionStart : null;

    mount.innerHTML =
      ledeView() +
      statsView() +
      controlsView(models.length) +
      listView(models) +
      (root.dataset.lsmColophon !== undefined ? colophonView() : "");

    if (caret !== null) {
      var input = mount.querySelector("#lsm-q");
      input.focus();
      input.setSelectionRange(caret, caret);
    }
    syncHash();
  }

  function toggleCategory(category) {
    var i = state.cats.indexOf(category);
    if (i === -1) state.cats.push(category);
    else state.cats.splice(i, 1);
  }

  function flash(button, label, ok) {
    var slot = button.querySelector(".lsm-ico-slot");
    var original = slot ? slot.innerHTML : button.innerHTML;
    if (slot) {
      if (ok) slot.innerHTML = icon("check");
    } else {
      button.innerHTML = html`${label}`;
    }
    button.dataset.state = ok ? "done" : "error";
    setTimeout(function () {
      if (slot) slot.innerHTML = original;
      else button.innerHTML = original;
      delete button.dataset.state;
    }, 1400);
  }

  // The async clipboard needs a secure context and a permission that some
  // browsers withhold; execCommand still works where it does not. If both
  // fail the text is left selected, so ⌘C has something to act on.
  function selectAndCopy(text, button) {
    var host = button.closest("dialog") || document.body;
    var field = document.createElement("textarea");
    field.value = text;
    field.setAttribute("readonly", "");
    field.style.cssText = "position:fixed;top:0;left:-9999px;opacity:0";
    host.appendChild(field);
    field.select();
    var ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (error) {
      ok = false;
    }
    if (ok) field.remove();
    else setTimeout(function () { field.remove(); }, 8000);
    return ok;
  }

  function copy(text, button, label) {
    var done = function () { flash(button, label, true); };
    var failed = function () {
      if (selectAndCopy(text, button)) done();
      else flash(button, "Press ⌘C", false);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, failed);
    } else {
      failed();
    }
  }

  function modelFor(key) {
    return data.models.find(function (m) { return m.keys.indexOf(key) !== -1; });
  }

  function onClick(event) {
    var hit = event.target.closest("[data-action], .lsm-copy-bib");
    if (!hit) return;
    var action = hit.dataset.action;

    if (hit.classList.contains("lsm-copy-bib")) {
      copy(panel._model.citation.bibtex, hit, "Copied");
      return;
    }
    if (action === "copykey") { copy(hit.dataset.key, hit, "Copied"); return; }
    if (action === "open") {
      var model = modelFor(hit.dataset.key);
      if (model) openPanel(model);
      return;
    }
    if (action === "close") { panel.close(); return; }
    if (action === "copy") { copy(panel._model.example, hit, "Copied"); return; }

    if (action === "cat") toggleCategory(hit.dataset.value);
    else if (action === "reset") state = { q: "", cats: [], gating: "", use: "", license: "" };
    else if (action) {
      state[action] = state[action] === hit.dataset.value ? "" : hit.dataset.value;
    }
    writeHash = true;
    render();
  }

  function onInput(event) {
    if (event.target.id !== "lsm-q") return;
    state.q = event.target.value;
    writeHash = true;
    clearTimeout(onInput._t);
    onInput._t = setTimeout(render, 120);
  }

  function onKey(event) {
    if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) return;
    var tag = (document.activeElement.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || document.activeElement.isContentEditable) return;
    var input = root.querySelector("#lsm-q");
    if (!input) return;
    event.preventDefault();
    input.focus();
    input.select();
  }

  /* ── theme ────────────────────────────────────────────────────── */

  function applyTheme() {
    var host = document.documentElement.dataset.theme;
    var theme;
    if (host === "dark" || host === "light") {
      theme = host;
    } else {
      theme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    root.dataset.lsmTheme = theme;
  }

  function watchTheme() {
    applyTheme();
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
    new MutationObserver(applyTheme).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"]
    });
  }

  /* ── boot ─────────────────────────────────────────────────────── */

  function ensureStylesheet() {
    var href = new URL("app.css", BASE).href;
    var already = Array.prototype.some.call(document.styleSheets, function (sheet) {
      return sheet.href === href;
    });
    if (already) return;
    var link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    document.head.appendChild(link);
  }

  function fail(mount, message) {
    mount.className = "lsm-root";
    mount.innerHTML = html`<div class="lsm-shell"><p class="lsm-empty">${message}
      <a href="${HOME}">Open the model zoo</a>.</p></div>`;
  }

  function start() {
    var mount = document.getElementById(MOUNT_ID);
    if (!mount) return;

    ensureStylesheet();
    root = mount;
    root.className = "lsm-root";
    embedded = root.dataset.lsmStandalone === undefined;
    if (embedded) root.dataset.lsmEmbedded = "";
    root.innerHTML = '<div class="lsm-shell"></div>';
    watchTheme();

    fetch(new URL("models.json", BASE).href)
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (payload) {
        data = payload;
        data.models.forEach(function (model) {
          model._hay = haystack(model);
          model._name = model.name.toLowerCase();
        });

        panel = document.createElement("dialog");
        panel.className = "lsm-panel lsm-root";
        panel.dataset.lsmTheme = root.dataset.lsmTheme;
        new MutationObserver(function () {
          panel.dataset.lsmTheme = root.dataset.lsmTheme;
        }).observe(root, { attributes: true, attributeFilter: ["data-lsm-theme"] });
        panel.addEventListener("click", function (event) {
          if (event.target === panel) panel.close();
        });
        panel.addEventListener("close", function () {
          if (lastFocused && lastFocused.isConnected) lastFocused.focus();
        });
        document.body.appendChild(panel);

        writeHash = readHash() !== null;
        render();

        root.addEventListener("click", onClick);
        panel.addEventListener("click", onClick);
        root.addEventListener("input", onInput);
        document.addEventListener("keydown", onKey);

        // #lsm-<key> opens that model directly — the shape a paper or an
        // issue can link to.
        var anchor = location.hash.match(/^#lsm-([\w.-]+)$/);
        if (anchor) {
          var target = modelFor(anchor[1]);
          if (target) openPanel(target);
        }
      })
      .catch(function (error) {
        fail(mount, "The model catalogue could not be loaded (" + error.message + ").");
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
