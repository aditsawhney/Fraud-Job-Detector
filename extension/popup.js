const VERDICT = {
    high:   { label: "LIKELY FRAUD" },
    medium: { label: "SUSPICIOUS"   },
    low:    { label: "LOOKS LEGITIMATE"  },
  };
  
  function render(result) {
    console.log("render called with:", result);
    if (!result) {
      document.getElementById("flags-list").innerHTML =
        `<div class="flag-placeholder">// no result yet — open a LinkedIn job post</div>`;
      return;
    }
  
    const { prediction, confidence, reasons, source, title, company } = result;
    const risk_level = (result.risk_level || "low").toLowerCase();
    
    if (result.unsupported) {
        document.getElementById("job-label").textContent = result.title || "—";
        document.getElementById("flags-list").innerHTML =
          `<div class="flag-placeholder">// open job in full view for analysis</div>`;
        return;
      }
      
    document.getElementById("job-label").textContent = company ? `${title} · ${company}` : title;
  
    const band = document.getElementById("risk-band");
    band.className = `risk-band ${risk_level}`;
  
    document.getElementById("risk-label").textContent  = VERDICT[risk_level].label;
    document.getElementById("risk-score").textContent  = `${Math.round(confidence * 100)}%`;
    // to this
    const confidenceLabel = risk_level === "high" ? "High confidence" : risk_level === "medium" ? "Medium confidence" : "Low confidence";
    document.getElementById("risk-verdict").textContent =
      `${confidenceLabel} · ${reasons.length} flag${reasons.length !== 1 ? "s" : ""} found`;
  
    const flagsList = document.getElementById("flags-list");
    if (reasons.length === 0) {
      flagsList.innerHTML = `<div class="flag-placeholder">// none detected</div>`;
    } else {
      flagsList.innerHTML = reasons.map((r) => {
        const severity = risk_level === "high" ? "red" : "amber";
        return `<div class="flag-item">
          <span class="flag-dot ${severity}"></span>${r}
        </div>`;
      }).join("");
    }
  
    const dot  = document.getElementById("status-dot");
    const text = document.getElementById("status-text");
    if (source === "model") {
      dot.className  = "status-dot ready";
      text.textContent = "Model Analysed";
    } else {
      dot.className  = "status-dot local";
      text.textContent = "Local Analysis";
    }
  }
  
  chrome.runtime.sendMessage({ type: "GET_RESULT" }, (result) => {
    if (result) render(result);
  });
  
  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === "ANALYSIS_RESULT") render(message.payload);
    if (message.type === "ANALYSIS_START") {
      document.getElementById("risk-band").className = "risk-band";
      document.getElementById("risk-score").textContent = "—";
      document.getElementById("risk-label").textContent = "scanning...";
      document.getElementById("risk-verdict").textContent = "waiting for result";
      document.getElementById("flags-list").innerHTML =
        `<div class="flag-placeholder">// analysing job posting</div>`;
    }
  });
  
  const poll = setInterval(() => {
    chrome.runtime.sendMessage({ type: "GET_RESULT" }, (result) => {
      if (result) {
        render(result);
        clearInterval(poll);
      }
    });
  }, 500);
  
  setTimeout(() => clearInterval(poll), 15000);