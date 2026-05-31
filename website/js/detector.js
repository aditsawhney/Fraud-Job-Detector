const API_URL = 'https://fraud-job-detector-1mdz.onrender.com/predict';

const EXAMPLES = {
  fraud: {
    title: 'Data Entry Operator – Work From Home',
    company: 'HR Solutions Consultancy Pvt Ltd',
    location: 'Work From Home (Pan India)',
    salary: '₹40,000 – ₹80,000 per month',
    contact: 'WhatsApp: +91 98765 43210 | hr.solutions2024@gmail.com',
    description: `Urgent Hiring! Limited seats available. Apply within 24 hours.

We are looking for freshers and experienced candidates for a simple data entry job from home. No experience required. Anyone can apply.

Earn ₹40,000–80,000 per month from the comfort of your home. Guaranteed salary. 100% placement assured.

Requirements:
- No qualification required
- Laptop/mobile required
- Training kit available (₹999 registration fee, refundable after 30 days)
- Immediate joiners preferred

Contact on WhatsApp only. Interview will be conducted via phone call.`,
  },
  legit: {
    title: 'Associate Software Engineer – Backend',
    company: 'ThoughtWorks India Pvt Ltd',
    location: 'Bangalore, Karnataka (Hybrid)',
    salary: '5–8 LPA',
    contact: 'careers@thoughtworks.com',
    description: `ThoughtWorks is looking for an Associate Software Engineer to join our backend team in Bangalore.

You'll work alongside senior engineers on real client projects, contributing to microservices built in Java and Go. We follow agile practices, pair programming, and continuous delivery.

Qualifications:
- B.E./B.Tech in CS or related field (2023/2024 batch)
- Proficiency in at least one backend language
- Familiarity with REST APIs and SQL

We offer competitive salaries, health insurance, 25 days paid leave, and learning budgets. We do not charge any fees at any stage of our recruitment process.`,
  },
};

function sanitize(str) {
    return str.replace(/[\u0000-\u001F\u007F]/g, ' ').trim();
  }

function loadExample(type) {
  const d = EXAMPLES[type];
  if (!d) return;
  document.getElementById('inp-title').value       = d.title;
  document.getElementById('inp-company').value     = d.company;
  document.getElementById('inp-location').value    = d.location;
  document.getElementById('inp-salary').value      = d.salary;
  document.getElementById('inp-contact').value     = d.contact;
  document.getElementById('inp-description').value = d.description;
}

function switchTab(tab, el) {
  document.querySelectorAll('.s2__tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.s2__panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
}

function riskClass(level) {
  if (!level) return 'warn';
  const l = level.toLowerCase();
  if (l.includes('high'))               return 'danger';
  if (l.includes('low') || l.includes('safe')) return 'safe';
  return 'warn';
}

function renderResult(data) {
  const { prediction, confidence, risk_level, reasons = [], top_model_features = [] } = data;
  const isFraud  = prediction === 'fraud' || prediction === 1 || prediction === '1';
  const rc       = riskClass(risk_level);
  const confPct  = Math.round((confidence || 0) * 100);
  const verdict  = isFraud ? 'Likely Fraudulent' : 'Looks Legitimate';
  const sub      = isFraud
    ? 'This posting matches known fraud patterns.'
    : 'No significant fraud signals detected.';

  const signals = reasons.length
    ? reasons.map(r => `<span class="signal-chip">${r}</span>`).join('')
    : `<span class="signal-chip signal-chip--none">No signals triggered</span>`;

    const features = top_model_features.length
    ? top_model_features.slice(0, 6).map(({ feature, coefficient }) => {
        const pct = Math.min(Math.round(Math.abs(coefficient) * 400), 100);
        return `
          <div class="result__feat-row">
            <span class="result__feat-name">${feature}</span>
            <div class="result__feat-bar-wrap">
              <div class="result__feat-bar" style="width:${pct}%"></div>
            </div>
            <span class="result__feat-val">${coefficient > 0 ? '+' : ''}${coefficient.toFixed(2)}</span>
          </div>`;
      }).join('')
    : '<p style="font-size:0.8rem;color:var(--text-muted)">Feature data unavailable</p>';

  return `
    <div class="result__header">
      <div>
        <div class="result__verdict" style="color:var(--${rc === 'danger' ? 'danger' : rc === 'safe' ? 'safe' : 'warn'})">${verdict}</div>
        <p class="result__sub">${sub}</p>
      </div>
      <div class="result__ring result__ring--${rc}">
        <span class="result__ring-val">${confPct}%</span>
        <span class="result__ring-lbl">conf</span>
      </div>
    </div>
    <div class="result__section">
      <div class="result__section-title">Detected signals</div>
      <div class="result__signals">${signals}</div>
    </div>
    <div class="result__section">
      <div class="result__section-title">Top model features</div>
      ${features}
    </div>
    <p style="font-size:0.75rem;color:var(--text-muted);margin-top:12px;padding-top:12px;border-top:1px solid var(--border)">
      Confidence reflects TF-IDF + regex signal agreement.
    </p>
  `;
}

function showLoading() {
  document.getElementById('result-idle').style.display    = 'none';
  document.getElementById('result-content').style.display = 'block';
  document.getElementById('result-content').innerHTML = `
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:380px;gap:12px;">
      <div class="spinner"></div>
      <p style="font-size:0.85rem;color:var(--text-muted)">Analysing…</p>
    </div>`;
}

async function analyse() {
    const body = {
        title:       sanitize(document.getElementById('inp-title')?.value       || ''),
        company:     sanitize(document.getElementById('inp-company')?.value     || ''),
        location:    sanitize(document.getElementById('inp-location')?.value    || ''),
        salary:      sanitize(document.getElementById('inp-salary')?.value      || ''),
        contact:     sanitize(document.getElementById('inp-contact')?.value     || ''),
        description: sanitize(document.getElementById('inp-description')?.value || ''),
      };

  if (!body.title && !body.description) {
    alert('Enter at least a title or description.');
    return;
  }

  // Switch to results tab automatically
  document.querySelectorAll('.s2__tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.s2__panel').forEach(p => p.classList.remove('active'));
  document.querySelector('.s2__tab').classList.add('active');
  document.getElementById('tab-results').classList.add('active');

  showLoading();

  try {
    const jsonStr = JSON.stringify(body);
    console.log('Sending:', jsonStr);
    
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: jsonStr,
    });

    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    console.log('Response:', JSON.stringify(data));
    document.getElementById('result-content').innerHTML = renderResult(data);
  } catch (err) {
    document.getElementById('result-content').innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:380px;gap:10px;text-align:center;padding:40px;">
        <p style="font-weight:500">API unreachable</p>
        <p style="font-size:0.82rem;color:var(--text-muted)">The Render backend may be cold-starting (~30s). Try again shortly.</p>
        <p style="font-size:0.75rem;font-family:var(--font-mono);color:var(--text-muted)">${err.message}</p>
      </div>`;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('analyze-btn')?.addEventListener('click', analyse);
});