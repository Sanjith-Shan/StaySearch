/* StaySearch demo UI: plain browser JavaScript, no build step. Talks to the service it is served from. */
(function () {
  "use strict";

  var API = ""; // same origin as `serve --static`
  var ROOM = ["Entire home/apt", "Private room", "Shared room", "Hotel room"];
  var STAGES = [
    ["query_tower", "Query tower", "var(--s1)"],
    ["probe", "IVF probe", "var(--s2)"],
    ["filter", "Lucene filter", "var(--s3)"],
    ["distance", "Exact distance", "var(--s4)"],
    ["rank", "Ranker", "var(--s5)"]
  ];

  var $ = function (id) { return document.getElementById(id); };
  var state = { meta: null, places: [], cities: [], guests: [], guestIdx: 0, markers: {}, activeId: null, areaBox: null, layer: null, boxLayer: null };

  // ---------------------------------------------------------------- helpers

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function getJson(path) {
    return fetch(API + path).then(function (r) {
      if (!r.ok) throw new Error(path + ": HTTP " + r.status);
      return r.json();
    });
  }

  function isoAdd(iso, days) {
    var d = new Date(iso + "T00:00:00Z");
    d.setUTCDate(d.getUTCDate() + days);
    return d.toISOString().slice(0, 10);
  }

  function fmtUs(us) {
    if (us == null) return "n/a";
    if (us >= 1000) return (us / 1000).toFixed(us >= 100000 ? 0 : 1) + " ms";
    return us + " µs";
  }

  function fmtPrice(p) {
    return p > 0 ? Math.round(p).toLocaleString() : "n/a";
  }

  function num(v, d) {
    return v == null || !isFinite(v) ? "n/a" : Number(v).toFixed(d);
  }

  function showError(msg) {
    var e = $("error");
    e.textContent = msg || "";
    e.hidden = !msg;
  }

  // ---------------------------------------------------------------- map

  var map = L.map("map", { zoomControl: true, attributionControl: true }).setView([40, -40], 3);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors | Data: Inside Airbnb, CC BY 4.0"
  }).addTo(map);
  map.attributionControl.setPrefix(false);
  state.layer = L.layerGroup().addTo(map);

  function boxBounds(b) {
    return L.latLngBounds([b.min_lat, b.min_lng], [b.max_lat, b.max_lng]);
  }

  function drawBox(b) {
    if (state.boxLayer) map.removeLayer(state.boxLayer);
    state.boxLayer = b ? L.rectangle(boxBounds(b), { color: "#0f766e", weight: 1.5, dashArray: "4 4", fill: true, fillOpacity: 0.03, interactive: false }).addTo(map) : null;
  }

  function placeBox(p) {
    // /places serializes the box as {min_lat, min_lng, max_lat, max_lng}; tolerate the flat bundle layout too.
    return p.box || { min_lat: p.min_lat, min_lng: p.min_lng, max_lat: p.max_lat, max_lng: p.max_lng };
  }

  // ---------------------------------------------------------------- pickers

  function selectedPlace() {
    var n = $("nbhd").value;
    var c = $("city").value;
    var idx = n !== "" ? Number(n) : (c !== "" ? Number(c) : null);
    return idx == null ? null : state.places[idx] || null;
  }

  function fillNbhds(keep) {
    var city = state.places[Number($("city").value)];
    var sel = $("nbhd");
    sel.innerHTML = "";
    sel.appendChild(new Option("Whole city", ""));
    if (!city) return;
    state.places
      .filter(function (p) { return p.kind === "nbhd" && p.city === city.city; })
      .sort(function (a, b) { return a.name.localeCompare(b.name); })
      .forEach(function (p) { sel.appendChild(new Option(p.name, String(p.idx))); });
    if (keep != null) sel.value = String(keep);
  }

  function focusPlace() {
    var p = selectedPlace();
    if (!p) return;
    state.areaBox = null;
    var b = placeBox(p);
    drawBox(b);
    map.fitBounds(boxBounds(b), { padding: [20, 20] });
  }

  function syncNprobe() {
    $("nprobe").disabled = $("mode").value !== "ivf";
  }

  // ---------------------------------------------------------------- search

  function buildRequest() {
    var req = {
      checkin: $("checkin").value,
      checkout: $("checkout").value,
      guests: Math.max(1, parseInt($("guests").value, 10) || 1),
      mode: $("mode").value,
      ranker: $("ranker").value,
      nprobe: Math.max(1, parseInt($("nprobe").value, 10) || 16),
      k: Math.max(1, parseInt($("k").value, 10) || 500),
      t: Math.max(1, parseInt($("t").value, 10) || 25),
      flexible: $("flexible").checked
    };
    var gid = $("guestId").value.trim();
    if (gid !== "") {
      if (!/^-?\d+$/.test(gid)) throw new Error("Guest id must be a whole number.");
      req.guest_id = gid; // sent as a string of digits so 64-bit ids keep their precision
    }
    if (!req.checkin || !req.checkout) throw new Error("Pick check-in and check-out dates.");
    if (req.checkout <= req.checkin) throw new Error("Check-out must be after check-in.");
    if (state.areaBox) {
      req.bbox = state.areaBox;
    } else {
      var p = selectedPlace();
      if (!p) throw new Error("Pick a city or search the map area.");
      req.place_idx = p.idx;
    }
    return req;
  }

  function search() {
    var req;
    try {
      req = buildRequest();
    } catch (e) {
      showError(e.message);
      return;
    }
    showError("");
    var btn = $("searchBtn");
    btn.disabled = true;
    btn.textContent = "Searching...";
    var t0 = performance.now();
    // guest_id goes out as a bare JSON number without passing through a JS double.
    var body = JSON.stringify(req).replace(/"guest_id":"(-?\d+)"/, '"guest_id":$1');
    fetch(API + "/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: body })
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (j) {
          if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
          return j;
        });
      })
      .then(function (resp) { render(req, resp, performance.now() - t0); })
      .catch(function (e) { showError("Search failed: " + e.message); })
      .then(function () {
        btn.disabled = false;
        btn.textContent = "Search";
      });
  }

  function chip(label, value, cls) {
    var c = el("span", "chip" + (cls ? " " + cls : ""));
    if (label) c.appendChild(document.createTextNode(label + " "));
    if (value != null) c.appendChild(el("b", null, String(value)));
    return c;
  }

  function renderSummary(req, resp, rttMs) {
    $("summary").hidden = false;
    var s = $("stats");
    s.innerHTML = "";
    s.appendChild(chip("shown", resp.results.length));
    s.appendChild(chip(resp.mode === "hnsw" ? "HNSW hits" : "passed filter", resp.candidates.toLocaleString()));
    s.appendChild(chip("retrieved", resp.retrieved.toLocaleString()));
    s.appendChild(chip("nights", resp.nights + (req.flexible ? " ±3d" : "")));
    s.appendChild(chip(resp.mode + (resp.mode === "ivf" ? " nprobe " + req.nprobe : "") + ", " + resp.ranker, null));
    if (resp.invariant_ok === true) {
      var ok = chip("✓ 0 results outside your dates, guests or area", null, "ok");
      ok.title = "The service re-checked every result night by night against the live availability bitset, min/max nights, guests and the map box.";
      s.appendChild(ok);
    } else if (resp.invariant_ok === false) {
      var bad = chip("✗ invariant violated: " + (resp.invariant_violations || []).length, null, "bad");
      bad.title = JSON.stringify(resp.invariant_violations || []);
      s.appendChild(bad);
    }

    var t = resp.timings_us || {};
    var sum = 0;
    STAGES.forEach(function (st) { sum += t[st[0]] || 0; });
    $("latTotal").textContent = "total " + fmtUs(t.total) + " \u00b7 round trip " + rttMs.toFixed(0) + " ms";
    var bar = $("latBar");
    var leg = $("latLegend");
    bar.innerHTML = "";
    leg.innerHTML = "";
    STAGES.forEach(function (st) {
      var v = t[st[0]] || 0;
      var seg = el("span");
      seg.style.background = st[2];
      seg.style.width = (sum > 0 ? (100 * v / sum) : 0) + "%";
      seg.title = st[1] + ": " + fmtUs(v);
      bar.appendChild(seg);
      var li = el("li");
      var sw = el("i");
      sw.style.background = st[2];
      li.appendChild(sw);
      li.appendChild(document.createTextNode(st[1]));
      li.appendChild(el("b", null, fmtUs(v)));
      leg.appendChild(li);
    });
  }

  function pinIcon(r, rank, active) {
    var label = r.price > 0 ? fmtPrice(r.price) : "#" + rank;
    return L.divIcon({
      className: "pin-wrap" + (active ? " active-wrap" : ""),
      html: '<span class="pin' + (active ? " active" : "") + '">' + label + "</span>",
      iconSize: null
    });
  }

  function setActive(id, pan) {
    var prev = state.activeId;
    state.activeId = id;
    [prev, id].forEach(function (k) {
      var m = state.markers[k];
      if (!m) return;
      m.marker.setIcon(pinIcon(m.r, m.rank, k === id));
      m.marker.setZIndexOffset(k === id ? 1000 : 0);
      if (m.item) m.item.classList.toggle("active", k === id);
    });
    var cur = state.markers[id];
    if (cur && pan) {
      map.panTo(cur.marker.getLatLng());
      cur.marker.openPopup();
    }
    if (cur && cur.item && !pan) cur.item.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function render(req, resp, rttMs) {
    renderSummary(req, resp, rttMs);
    state.layer.clearLayers();
    state.markers = {};
    state.activeId = null;
    var list = $("list");
    list.innerHTML = "";
    $("empty").hidden = resp.results.length > 0;

    var bounds = [];
    resp.results.forEach(function (r, i) {
      var rank = i + 1;
      var room = ROOM[r.room_type] || "Room type " + r.room_type;
      var marker = L.marker([r.lat, r.lng], { icon: pinIcon(r, rank, false), title: "#" + rank + " " + room, riseOnHover: true });
      var pop = el("div");
      pop.appendChild(el("div", "pop-t", "#" + rank + " · " + room));
      pop.appendChild(el("div", "pop-m", fmtPrice(r.price) + " per night · up to " + r.accommodates + " guests"));
      pop.appendChild(el("div", "pop-m", "listing " + r.id + " · " + r.min_nights + "-" + r.max_nights + " nights"));
      marker.bindPopup(pop);
      marker.on("click", function () { setActive(r.id, false); });
      marker.addTo(state.layer);
      bounds.push([r.lat, r.lng]);

      var li = el("li", "item");
      li.tabIndex = 0;
      li.appendChild(el("span", "rank", String(rank)));
      var mid = el("div");
      mid.appendChild(el("div", "title", room + " · " + r.accommodates + " guests"));
      var parts = ["dist " + num(r.distance, 3)];
      if (r.score != null) parts.push("score " + num(r.score, 3));
      parts.push(r.min_nights + "-" + (r.max_nights >= 1000 ? "∞" : r.max_nights) + " nights");
      mid.appendChild(el("div", "meta", parts.join(" · ")));
      li.appendChild(mid);
      var price = el("div", "price", fmtPrice(r.price));
      price.appendChild(el("small", null, r.price > 0 ? "per night" : "no quote"));
      li.appendChild(price);
      li.addEventListener("click", function () { setActive(r.id, true); });
      li.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setActive(r.id, true); } });
      list.appendChild(li);

      state.markers[r.id] = { marker: marker, item: li, r: r, rank: rank };
    });

    if (!state.areaBox) {
      var p = state.places[resp.place_idx];
      if (req.place_idx != null && p) drawBox(placeBox(p));
    }
    if (bounds.length > 0 && !state.areaBox) map.fitBounds(L.latLngBounds(bounds), { padding: [40, 40], maxZoom: 15 });
  }

  // ---------------------------------------------------------------- sample guests

  function useSampleGuest() {
    if (!state.guests.length) {
      showError("This bundle has no test guests with history.");
      return;
    }
    var g = state.guests[state.guestIdx % state.guests.length];
    state.guestIdx++;
    $("guestId").value = String(g.guest_id);
    var p = state.places[g.place_idx];
    if (p) {
      var cityIdx = p.kind === "city" ? p.idx : (state.cities.find(function (c) { return c.city === p.city; }) || {}).idx;
      if (cityIdx != null) {
        $("city").value = String(cityIdx);
        fillNbhds(p.kind === "nbhd" ? p.idx : "");
      }
      focusPlace();
    }
    $("checkin").value = g.checkin;
    $("checkout").value = g.checkout;
    $("guests").value = g.guests;
    var note = $("sampleNote");
    note.textContent = "Guest " + g.guest_id + " with history, from a test query: " + (g.place_name || "place " + g.place_idx) +
      ", " + g.checkin + " to " + g.checkout + ", " + g.guests + " guest" + (g.guests === 1 ? "" : "s") + ".";
    note.hidden = false;
  }

  // ---------------------------------------------------------------- init

  function initMeta(meta) {
    state.meta = meta;
    var status = $("status");
    status.textContent = meta.n_listings.toLocaleString() + " listings · snapshot " + meta.as_of + (meta.synthetic ? " · SYNTHETIC bundle" : "");
    status.classList.toggle("synthetic", !!meta.synthetic);

    var lo = meta.base_date;
    var hi = isoAdd(meta.calendar_last_night, 1);
    ["checkin", "checkout"].forEach(function (id) { $(id).min = lo; $(id).max = hi; });
    var ci = isoAdd(meta.as_of, 14);
    if (ci > meta.calendar_last_night) ci = meta.as_of;
    $("checkin").value = ci;
    $("checkout").value = isoAdd(ci, 3);

    var mode = $("mode");
    meta.modes.forEach(function (m) { mode.appendChild(new Option(m.toUpperCase(), m)); });
    mode.value = meta.defaults.mode;
    var rk = $("ranker");
    meta.rankers.forEach(function (r) { rk.appendChild(new Option(r === "none" ? "none (distance)" : r.toUpperCase(), r)); });
    rk.value = meta.defaults.ranker;
    $("nprobe").value = meta.defaults.nprobe;
    if (meta.defaults.k) $("k").value = meta.defaults.k;
    $("nprobe").max = meta.n_clusters;
    syncNprobe();
  }

  function initPlaces(places) {
    state.places = places;
    state.cities = places.filter(function (p) { return p.kind === "city"; }).sort(function (a, b) { return a.name.localeCompare(b.name); });
    var sel = $("city");
    state.cities.forEach(function (c) { sel.appendChild(new Option(c.name, String(c.idx))); });
    if (state.cities.length) {
      sel.value = String(state.cities[0].idx);
      fillNbhds();
      focusPlace();
    }
  }

  $("city").addEventListener("change", function () { fillNbhds(); focusPlace(); });
  $("nbhd").addEventListener("change", focusPlace);
  $("mode").addEventListener("change", syncNprobe);
  $("sampleBtn").addEventListener("click", useSampleGuest);
  $("checkin").addEventListener("change", function () {
    if ($("checkout").value <= $("checkin").value) $("checkout").value = isoAdd($("checkin").value, 1);
  });
  $("areaBtn").addEventListener("click", function () {
    var b = map.getBounds();
    state.areaBox = { min_lat: b.getSouth(), min_lng: b.getWest(), max_lat: b.getNorth(), max_lng: b.getEast() };
    drawBox(null);
    search();
  });
  $("form").addEventListener("submit", function (e) {
    e.preventDefault();
    state.areaBox = null;
    search();
  });

  Promise.all([getJson("/meta"), getJson("/places"), getJson("/sample-guests").catch(function () { return []; })])
    .then(function (r) {
      initMeta(r[0]);
      initPlaces(r[1]);
      state.guests = r[2] || [];
      $("sampleBtn").disabled = state.guests.length === 0;
      search();
    })
    .catch(function (e) {
      $("status").textContent = "service unreachable";
      showError("Could not load /meta or /places: " + e.message);
    });
})();
