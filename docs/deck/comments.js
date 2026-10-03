/* Deck Comments — Google-Docs-style commenting on HTML slides.
 * (Adapted from the letter-comments UI: anchors also record the slide, its title
 * and the highlighted element's full text, so "address comments" has context.)
 *
 * Usage:
 *   <link rel="stylesheet" href="comments.css">
 *   <script src="comments.js" defer></script>
 *
 * Features:
 *   - Select text → "+ comment" floats next to the selection → click → modal
 *   - Comments persist in localStorage (scoped per filename)
 *   - Export to JSON (file: comments-<basename>.json) for Claude to process
 *   - Import JSON to load comments from another machine/iteration
 *   - Resolve / reopen / delete from the side panel
 *   - Click a comment → scrolls to and flashes its location
 *   - Hidden in print (so PDF export is clean)
 */
(function () {
  'use strict';

  var DOC_ID = (location.pathname.split('/').pop() || 'index.html');
  var STORAGE_KEY = 'letter-comments::' + DOC_ID;
  var SCHEMA_VERSION = 1;
  var CONTEXT_LEN = 50;
  // The Python letter-server provides /api/comments/<doc>; available only when
  // the page is loaded via http(s) on localhost. file:// loads fall back to
  // localStorage so the UI still works without the server.
  var SERVER_AVAILABLE = (location.protocol === 'http:' || location.protocol === 'https:') &&
                         (location.hostname === 'localhost' || location.hostname === '127.0.0.1');
  var SERVER_HEALTHY = SERVER_AVAILABLE; // flips to false on first failed POST
  var SERVER_URL = '/api/comments/' + DOC_ID;

  var comments = [];

  function buildPayload() {
    return {
      doc: DOC_ID,
      schema_version: SCHEMA_VERSION,
      exported_at: new Date().toISOString(),
      comments: comments
    };
  }

  // ── Storage ────────────────────────────────
  function loadLocal() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      var data = JSON.parse(raw);
      if (Array.isArray(data.comments)) comments = data.comments;
    } catch (e) {
      console.warn('comments local load failed', e);
    }
  }
  function persistLocal() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(buildPayload())); }
    catch (e) { console.warn('comments local save failed', e); }
  }
  function persistServer() {
    if (!SERVER_HEALTHY) return Promise.resolve(false);
    return fetch(SERVER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPayload())
    }).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      flashStatus('saved');
      return true;
    }).catch(function (e) {
      console.warn('comments autosave to server failed', e);
      SERVER_HEALTHY = false;
      flashStatus('offline (using localStorage)');
      return false;
    });
  }
  function persist() {
    persistLocal();
    persistServer();
  }
  function loadServer() {
    if (!SERVER_AVAILABLE) return Promise.resolve(false);
    return fetch(SERVER_URL, { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function (data) {
      if (data && Array.isArray(data.comments)) {
        comments = data.comments;
        persistLocal(); // mirror to localStorage as a recovery copy
        return true;
      }
      return false;
    }).catch(function (e) {
      SERVER_HEALTHY = false;
      console.warn('comments load from server failed; falling back to localStorage', e);
      return false;
    });
  }
  function flashStatus(text) {
    var el = document.querySelector('#comments-toolbar .status');
    if (!el) return;
    el.textContent = text;
    el.style.opacity = '1';
    clearTimeout(flashStatus._t);
    flashStatus._t = setTimeout(function () { el.style.opacity = '0'; }, 1200);
  }

  // ── JSON export / import ───────────────────
  function exportJSON() {
    var payload = {
      doc: DOC_ID,
      schema_version: SCHEMA_VERSION,
      exported_at: new Date().toISOString(),
      comments: comments
    };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'comments-' + DOC_ID.replace(/\.html?$/, '') + '.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }
  function importJSON() {
    var input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,application/json';
    input.onchange = function (e) {
      var file = e.target.files && e.target.files[0];
      if (!file) return;
      var reader = new FileReader();
      reader.onload = function (ev) {
        try {
          var data = JSON.parse(ev.target.result);
          if (!Array.isArray(data.comments)) {
            alert('JSON does not contain a "comments" array.');
            return;
          }
          comments = data.comments;
          persist();
          renderHighlights();
          renderPanel();
          updateCount();
        } catch (err) {
          alert('Failed to parse JSON: ' + err.message);
        }
      };
      reader.readAsText(file);
    };
    input.click();
  }

  // ── Anchor extraction (uses plain-text body) ──
  function closestEl(node, selector) {
    var el = node && (node.nodeType === 1 ? node : node.parentElement);
    return el && el.closest ? el.closest(selector) : null;
  }
  function extractAnchor(range) {
    var quote = range.toString().trim();
    if (!quote) return null;
    var fullText = getDocText();
    // Prefer the occurrence that starts inside the selected node (quotes can repeat across slides).
    var nodes = collectTextNodes(), idx = -1;
    for (var i = 0; i < nodes.length; i++) {
      if (nodes[i].node === range.startContainer) {
        var guess = nodes[i].start + range.startOffset;
        idx = fullText.indexOf(quote, Math.max(0, guess - 2));
        if (idx > guess + 2) idx = -1;
        break;
      }
    }
    if (idx < 0) idx = fullText.indexOf(quote);
    if (idx < 0) return null;
    var slide = closestEl(range.startContainer, '.slide');
    var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
    var block = closestEl(range.startContainer, 'p, li, h1, h2, h3, .ln, .file, .bar, .sub, .body, text, figcaption, .eyebrow') ||
                closestEl(range.startContainer, 'div');
    return {
      quote: quote,
      prefix: fullText.substring(Math.max(0, idx - CONTEXT_LEN), idx),
      suffix: fullText.substring(idx + quote.length, idx + quote.length + CONTEXT_LEN),
      slide: slide ? slides.indexOf(slide) + 1 : null,
      slide_title: slide ? (slide.getAttribute('aria-label') || '') : '',
      element_text: block ? block.textContent.replace(/\s+/g, ' ').trim().slice(0, 400) : '',
      in_diagram: !!closestEl(range.startContainer, 'svg')
    };
  }

  // Build flat text representation of the body, skipping our own UI nodes.
  function getDocText() {
    var nodes = collectTextNodes();
    return nodes.map(function (n) { return n.text; }).join('');
  }
  function collectTextNodes() {
    var nodes = [];
    var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        var p = node.parentElement;
        while (p) {
          var id = p.id;
          if (id === 'comments-toolbar' || id === 'comments-panel' ||
              id === 'comments-floating-btn' || id === 'comment-modal' ||
              (p.classList && p.classList.contains('deck-nav'))) {
            return NodeFilter.FILTER_REJECT;
          }
          p = p.parentElement;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    var n;
    var offset = 0;
    while ((n = walker.nextNode())) {
      nodes.push({ node: n, text: n.nodeValue, start: offset });
      offset += n.nodeValue.length;
    }
    return nodes;
  }

  // ── CRUD ───────────────────────────────────
  function addComment(anchor, text) {
    comments.push({
      id: 'c-' + Date.now() + '-' + Math.random().toString(36).slice(2, 7),
      created_at: new Date().toISOString(),
      status: 'open',
      anchor: anchor,
      comment: text
    });
    persist();
    renderHighlights();
    renderPanel();
    updateCount();
  }
  function deleteComment(id) {
    comments = comments.filter(function (c) { return c.id !== id; });
    persist();
    renderHighlights();
    renderPanel();
    updateCount();
  }
  function toggleResolve(id) {
    var c = comments.find(function (x) { return x.id === id; });
    if (!c) return;
    c.status = c.status === 'resolved' ? 'open' : 'resolved';
    if (c.status === 'resolved') c.resolved_at = new Date().toISOString();
    else delete c.resolved_at;
    persist();
    renderHighlights();
    renderPanel();
    updateCount();
  }

  // ── Rendering: highlights in document ─────
  function clearHighlights() {
    var svgHits = document.querySelectorAll('.comment-svg-highlight');
    for (var j = 0; j < svgHits.length; j++) {
      svgHits[j].classList.remove('comment-svg-highlight', 'resolved');
      delete svgHits[j].dataset.commentId;
    }
    var hits = document.querySelectorAll('.comment-highlight');
    for (var i = 0; i < hits.length; i++) {
      var el = hits[i];
      var parent = el.parentNode;
      while (el.firstChild) parent.insertBefore(el.firstChild, el);
      parent.removeChild(el);
      parent.normalize();
    }
  }
  function renderHighlights() {
    clearHighlights();
    comments.forEach(function (c) {
      if (!c.anchor || !c.anchor.quote) return;
      highlightQuote(c.anchor, c.id, c.status);
    });
  }
  function highlightQuote(anchor, id, status) {
    var nodes = collectTextNodes();
    var fullText = nodes.map(function (n) { return n.text; }).join('');
    // Disambiguate using prefix + quote + suffix
    var search = (anchor.prefix || '') + anchor.quote + (anchor.suffix || '');
    var idx = -1;
    if (search.length > anchor.quote.length) {
      idx = fullText.indexOf(search);
      if (idx >= 0) idx += (anchor.prefix || '').length;
    }
    if (idx < 0) idx = fullText.indexOf(anchor.quote);
    if (idx < 0) return;
    var endIdx = idx + anchor.quote.length;
    var startNode = null, startOffset = 0, endNode = null, endOffset = 0;
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      if (!startNode && idx >= n.start && idx < n.start + n.text.length) {
        startNode = n.node;
        startOffset = idx - n.start;
      }
      if (endIdx > n.start && endIdx <= n.start + n.text.length) {
        endNode = n.node;
        endOffset = endIdx - n.start;
        break;
      }
    }
    if (!startNode || !endNode) return;
    // Text inside SVG (diagram labels): an HTML span there would make the text
    // vanish, so mark the <text> element instead.
    var svgText = closestEl(startNode, 'text');
    if (svgText && closestEl(startNode, 'svg')) {
      svgText.classList.add('comment-svg-highlight');
      if (status === 'resolved') svgText.classList.add('resolved');
      svgText.dataset.commentId = id;
      return;
    }
    try {
      var range = document.createRange();
      range.setStart(startNode, startOffset);
      range.setEnd(endNode, endOffset);
      var span = document.createElement('span');
      span.className = 'comment-highlight' + (status === 'resolved' ? ' resolved' : '');
      span.dataset.commentId = id;
      span.title = 'Click to view this comment';
      span.addEventListener('click', function (e) {
        e.stopPropagation();
        scrollToComment(id);
      });
      range.surroundContents(span);
    } catch (e) {
      // Range crosses block boundaries — can't surround in one span. Skip.
      console.warn('Could not highlight quote (likely crosses block boundary):',
                   anchor.quote.substring(0, 60));
    }
  }

  // ── Rendering: side panel ─────────────────
  function renderPanel() {
    var list = document.getElementById('comments-list');
    if (!list) return;
    if (!comments.length) {
      list.innerHTML = '<div style="color:#888;text-align:center;padding:24px 12px;font-style:italic;">No comments yet.<br><br>Select any text on a slide and click <b>+ comment</b>.</div>';
      return;
    }
    // Sort: open first, then by created_at desc
    var sorted = comments.slice().sort(function (a, b) {
      if (a.status === b.status) return (b.created_at || '').localeCompare(a.created_at || '');
      return a.status === 'resolved' ? 1 : -1;
    });
    list.innerHTML = '';
    sorted.forEach(function (c) {
      var item = document.createElement('div');
      item.className = 'comment-item' + (c.status === 'resolved' ? ' resolved' : '');
      item.dataset.commentId = c.id;
      item.addEventListener('click', function () { scrollToComment(c.id); });

      var quote = document.createElement('div');
      quote.className = 'quote';
      var q = (c.anchor && c.anchor.quote) || '(no quote)';
      var where = c.anchor && c.anchor.slide ? 'Slide ' + c.anchor.slide + ' · ' : '';
      quote.textContent = where + (q.length > 90 ? q.substring(0, 90) + '…' : q);
      item.appendChild(quote);

      var text = document.createElement('div');
      text.className = 'text';
      text.textContent = c.comment;
      item.appendChild(text);

      var meta = document.createElement('div');
      meta.className = 'meta';
      var when = new Date(c.created_at || Date.now()).toLocaleString();
      var status = c.status || 'open';
      meta.innerHTML = '<span>' + when + ' · ' + status + '</span>';
      var actions = document.createElement('span');
      var resolveBtn = document.createElement('button');
      resolveBtn.textContent = c.status === 'resolved' ? 'Reopen' : 'Resolve';
      resolveBtn.addEventListener('click', function (e) { e.stopPropagation(); toggleResolve(c.id); });
      actions.appendChild(resolveBtn);
      var delBtn = document.createElement('button');
      delBtn.textContent = 'Delete';
      delBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        if (confirm('Delete this comment?')) deleteComment(c.id);
      });
      actions.appendChild(delBtn);
      meta.appendChild(actions);
      item.appendChild(meta);

      list.appendChild(item);
    });
  }

  function scrollToComment(id) {
    var c = comments.find(function (x) { return x.id === id; });
    if (c && c.anchor && c.anchor.slide && window.deckShow) window.deckShow(c.anchor.slide - 1);
    var el = document.querySelector('[data-comment-id="' + id + '"]:not(.comment-item)');
    if (!el) return;
    var orig = el.style.backgroundColor;
    el.style.transition = 'background-color 0.3s';
    el.style.backgroundColor = '#ffe082';
    setTimeout(function () { el.style.backgroundColor = orig; }, 900);
  }

  function updateCount() {
    var open = comments.filter(function (c) { return c.status !== 'resolved'; }).length;
    var total = comments.length;
    var cnt = document.querySelector('#comments-toolbar .count');
    if (!cnt) return;
    cnt.textContent = total === 0 ? '' : (open + (total > open ? ' / ' + total : ''));
    cnt.style.display = total > 0 ? 'inline-block' : 'none';
  }

  // ── UI: floating "+ comment" button on text selection ──
  var floatingBtn;
  function ensureFloatingBtn() {
    if (floatingBtn) return floatingBtn;
    floatingBtn = document.createElement('button');
    floatingBtn.id = 'comments-floating-btn';
    floatingBtn.textContent = '+ comment';
    floatingBtn.style.display = 'none';
    floatingBtn.addEventListener('mousedown', function (e) { e.preventDefault(); });
    floatingBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      var sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) return;
      var range = sel.getRangeAt(0);
      var anchor = extractAnchor(range);
      if (!anchor) return;
      // Wrap the selection in a temporary highlight so the user can SEE
      // what they're commenting on while the modal is open. On Save the
      // renderHighlights pass will replace it with a permanent .comment-highlight;
      // on Cancel we remove it.
      var tempSpan = wrapRangeInTempHighlight(range);
      sel.removeAllRanges();
      hideFloatingBtn();
      openModal(anchor, tempSpan);
    });
    document.body.appendChild(floatingBtn);
    return floatingBtn;
  }

  function wrapRangeInTempHighlight(range) {
    if (closestEl(range.startContainer, 'svg')) return null;  // see highlightQuote
    try {
      var span = document.createElement('span');
      span.className = 'comment-highlight pending';
      range.surroundContents(span);
      return span;
    } catch (e) {
      // Range crosses block boundaries — can't wrap in a single span. Skip silently.
      return null;
    }
  }

  function removeTempHighlight(span) {
    if (!span || !span.parentNode) return;
    var parent = span.parentNode;
    while (span.firstChild) parent.insertBefore(span.firstChild, span);
    parent.removeChild(span);
    parent.normalize();
  }
  function showFloatingBtn(rect) {
    var btn = ensureFloatingBtn();
    btn.style.left = (window.scrollX + rect.right + 6) + 'px';
    btn.style.top = (window.scrollY + Math.max(0, rect.top - 4)) + 'px';
    btn.style.display = 'block';
  }
  function hideFloatingBtn() {
    if (floatingBtn) floatingBtn.style.display = 'none';
  }
  function selectionInsideUI(sel) {
    var node = sel.anchorNode;
    while (node) {
      if (node.id === 'comments-toolbar' || node.id === 'comments-panel' ||
          node.id === 'comment-modal' || node.id === 'comments-floating-btn') return true;
      node = node.parentElement;
    }
    return false;
  }
  document.addEventListener('mouseup', function () {
    setTimeout(function () {
      var sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) { hideFloatingBtn(); return; }
      var text = sel.toString().trim();
      if (text.length < 3) { hideFloatingBtn(); return; }
      if (selectionInsideUI(sel)) { hideFloatingBtn(); return; }
      var rect = sel.getRangeAt(0).getBoundingClientRect();
      if (rect && rect.width > 0) showFloatingBtn(rect);
    }, 10);
  });
  document.addEventListener('mousedown', function (e) {
    if (e.target && e.target.id !== 'comments-floating-btn') hideFloatingBtn();
  });

  // ── UI: modal ─────────────────────────────
  function openModal(anchor, tempSpan) {
    var modal = document.getElementById('comment-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'comment-modal';
      modal.innerHTML =
        '<div class="quote"></div>' +
        '<textarea placeholder="Write your comment…"></textarea>' +
        '<div class="actions">' +
        '  <button class="cancel">Cancel</button>' +
        '  <button class="save">Save</button>' +
        '</div>';
      document.body.appendChild(modal);
    }
    modal.querySelector('.quote').textContent = anchor.quote;
    var ta = modal.querySelector('textarea');
    ta.value = '';
    modal.classList.add('open');
    setTimeout(function () { ta.focus(); }, 50);
    function cleanupAndClose() {
      modal.classList.remove('open');
      if (tempSpan) removeTempHighlight(tempSpan);
    }
    function commitAndClose() {
      var text = ta.value.trim();
      if (!text) { ta.focus(); return; }
      // Remove the temp span before addComment → renderHighlights so the
      // permanent highlight cleanly replaces it without flicker.
      if (tempSpan) removeTempHighlight(tempSpan);
      addComment(anchor, text);
      modal.classList.remove('open');
    }
    modal.querySelector('.cancel').onclick = cleanupAndClose;
    modal.querySelector('.save').onclick = commitAndClose;
    ta.onkeydown = function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        commitAndClose();
      } else if (e.key === 'Escape') {
        cleanupAndClose();
      }
    };
  }

  // ── UI: toolbar + panel skeleton ──────────
  function buildToolbar() {
    var tb = document.createElement('div');
    tb.id = 'comments-toolbar';
    tb.innerHTML =
      '<span class="status"></span>' +
      '<span class="count" style="display:none">0</span>' +
      '<button id="ct-toggle">Comments</button>' +
      '<button id="ct-export">Export JSON</button>' +
      '<button id="ct-import">Import JSON</button>';
    document.body.appendChild(tb);
    document.getElementById('ct-toggle').addEventListener('click', togglePanel);
    document.getElementById('ct-export').addEventListener('click', exportJSON);
    document.getElementById('ct-import').addEventListener('click', importJSON);
    if (!SERVER_AVAILABLE) {
      flashStatus('localStorage only — run docs/deck/serve.py for autosave');
      var s = document.querySelector('#comments-toolbar .status');
      if (s) s.style.opacity = '1'; // keep visible since this is a persistent state
    }
  }
  function buildPanel() {
    var p = document.createElement('div');
    p.id = 'comments-panel';
    p.innerHTML =
      '<header>' +
      '  <h3>Comments</h3>' +
      '  <span class="actions"><button id="cp-close" title="Close">×</button></span>' +
      '</header>' +
      '<div id="comments-list"></div>';
    document.body.appendChild(p);
    document.getElementById('cp-close').addEventListener('click', togglePanel);
  }
  function togglePanel() {
    var panel = document.getElementById('comments-panel');
    panel.classList.toggle('open');
    if (panel.classList.contains('open')) renderPanel();
  }

  // ── Init ──────────────────────────────────
  function init() {
    buildToolbar();
    buildPanel();
    // Try to load from server first; if that fails (or we're on file://),
    // load from localStorage. Either way, render highlights when done.
    if (SERVER_AVAILABLE) {
      loadServer().then(function (ok) {
        if (!ok) loadLocal();
        renderHighlights();
        updateCount();
        renderPanel();
      });
    } else {
      loadLocal();
      renderHighlights();
      updateCount();
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
