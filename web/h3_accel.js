/**
 * H3 Accel Selector -- live filtering of the acceleration-LoRA dropdown.
 *
 * The backend knows every acceleration file on disk and can judge each one
 * against a base; what it cannot know at request time is WHICH base the
 * workflow loaded, because that lives in the upstream loader node's widget.
 * So the frontend walks upstream from the picker's MODEL input, finds the
 * checkpoint filename, and asks the backend to re-filter.
 *
 * The backend still verifies the pairing at execution time against the live
 * weights -- this dropdown is a convenience, the runtime check is the
 * guarantee.
 */

import { api } from "../../scripts/api.js";

const PICKER = "H3AccelLoRAPicker";
const SCAN_URL = "/h3_accel/scan";
const REFRESH_MS = 400;

function setComboOptions(widget, values) {
    if (!widget) return;
    if (widget.options && Array.isArray(widget.options.values)) {
        widget.options.values = values;
        if (Array.isArray(widget.options.realValues)) widget.options.realValues = values;
    } else if (Array.isArray(widget.options)) {
        widget.options = values;
    } else {
        widget.options = values;
    }
    if (values.length && !values.includes(widget.value)) {
        widget.value = values[0];
    }
    widget.callback?.(widget.value);
}

function widgetValue(node, name) {
    const w = (node.widgets || []).find((x) => x.name === name);
    return w ? w.value : undefined;
}

/** Walk upstream from the picker's MODEL input to the checkpoint filename. */
function findUpstreamBase(node) {
    const graph = app.graph;
    if (!graph) return null;
    const seen = new Set();
    const queue = [];

    const modelInput = (node.inputs || []).find((i) => i.type === "MODEL");
    if (modelInput && modelInput.link != null) {
        const link = graph.links[modelInput.link];
        if (link) queue.push(graph.getNodeById(link.origin_id));
    }

    const HINTS = /minimax|fastvideo|(^|[^a-z])h3([^a-z]|$)/i;
    while (queue.length) {
        const n = queue.shift();
        if (!n || seen.has(n.id)) continue;
        seen.add(n.id);
        for (const w of n.widgets || []) {
            if (typeof w.value !== "string") continue;
            if (!/\.safetensors$/i.test(w.value)) continue;
            if (HINTS.test(w.value)) return w.value;
        }
        for (const inp of n.inputs || []) {
            if (inp.type === "MODEL" && inp.link != null) {
                const link = graph.links[inp.link];
                if (link) queue.push(graph.getNodeById(link.origin_id));
            }
        }
    }
    return null;
}

async function refreshPicker(node) {
    const baseName = findUpstreamBase(node);
    const hintWidget = (node.widgets || []).find((w) => w.name === "base_hint");
    if (hintWidget) hintWidget.value = baseName || "";

    const showAll = !!widgetValue(node, "show_incompatible");
    let payload = { base_name: baseName || "", show_incompatible: showAll };
    try {
        const res = await api.fetchApi(SCAN_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        const items = data?.items || [];
        const values = items.map((i) => i.key);
        const combo = (node.widgets || []).find((w) => w.name === "accel_lora");
        if (!combo) return;
        if (!values.length) {
            setComboOptions(combo, ["(No matching acceleration LoRA for this base)"]);
            combo.disabled = true;
            return;
        }
        combo.disabled = false;
        setComboOptions(combo, values);
        node.setDirtyCanvas?.(true, true);
    } catch (err) {
        console.warn("[H3 Accel] scan failed:", err);
    }
}

app.registerExtension({
    name: "BSAI.H3AccelSelector",

    async nodeCreated(node) {
        if (node.type !== PICKER) return;
        const trigger = () => {
            clearTimeout(node.__h3AccelTimer);
            node.__h3AccelTimer = setTimeout(() => refreshPicker(node), REFRESH_MS);
        };
        node.__h3AccelRefresh = trigger;

        // Repopulate as soon as the node exists, and again whenever the
        // upstream model link or the "show incompatible" switch changes.
        trigger();
        setTimeout(trigger, 900);

        const origCallback = node.onWidgetChanged;
        node.onWidgetChanged = function (w) {
            if (w?.name === "show_incompatible") trigger();
            return origCallback?.apply(this, arguments);
        };
    },

    async setup() {
        const debounced = () => {
            clearTimeout(app.__h3AccelTimer);
            app.__h3AccelTimer = setTimeout(() => {
                for (const n of app.graph?._nodes || []) {
                    if (n.type === PICKER) n.__h3AccelRefresh?.();
                }
            }, REFRESH_MS);
        };
        app.graph?.on("node:connected", debounced);
        app.graph?.on("node:removed", debounced);
        app.graph?.on("change", debounced);
    },
});
