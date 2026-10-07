/**
 * ChurnVision AI Application Logic & Chart Controllers
 */

document.addEventListener('DOMContentLoaded', async function () {
  // Chart instances
  let donutChart = null;
  let contractChart = null;
  let supportChart = null;
  let tenureScatterChart = null;
  let modelCompareChart = null;
  let featureImpChart = null;
  let cmDonutChart = null;

  let currentBatchData = [];
  let modelResults = null;

  // Code snippets cache
  const codeSnippets = {
    train: `import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n\n# 1. Load Data\ndf = pd.read_csv('telecom_churn.csv')\ny = (df['Churn'] == 'Yes').astype(int)\nX = df.drop(columns=['customerID', 'Churn'])\nX_encoded = pd.get_dummies(X, drop_first=True)\n\n# 2. Train Test Split & Scale\nX_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)\n\n# 3. Model Training & Benchmarks\nmodels = {\n    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),\n    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),\n    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)\n}\n\nfor name, model in models.items():\n    if name == "Logistic Regression":\n        model.fit(X_train_scaled, y_train)\n        preds = model.predict(X_test_scaled)\n    else:\n        model.fit(X_train, y_train)\n        preds = model.predict(X_test)\n    \n    print(f"[{name}] Acc: {accuracy_score(y_test, preds):.4f} | F1: {f1_score(y_test, preds):.4f}")`,
    gen: `import pandas as pd\nimport numpy as np\n\ndef generate_churn_dataset(n_samples=5000):\n    np.random.seed(42)\n    tenure = np.random.randint(1, 73, size=n_samples)\n    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples)\n    monthly_charges = np.round(np.random.uniform(18.5, 118.9, size=n_samples), 2)\n    \n    logit = -0.5 - 0.05*tenure + 1.8*(contract=='Month-to-month') + 0.015*(monthly_charges-60)\n    prob = 1.0 / (1.0 + np.exp(-logit))\n    churn = np.where(np.random.rand(n_samples) < prob, 'Yes', 'No')\n    \n    return pd.DataFrame({'tenure': tenure, 'Contract': contract, 'MonthlyCharges': monthly_charges, 'Churn': churn})\n\ndf = generate_churn_dataset()\ndf.to_csv('telecom_churn.csv', index=False)`,
    pred: `import numpy as np\n\ndef predict_churn(customer_data):\n    tenure = float(customer_data.get('tenure', 12))\n    contract = customer_data.get('Contract', 'Month-to-month')\n    monthly = float(customer_data.get('MonthlyCharges', 65.0))\n    \n    logit = -0.5 - 0.045*tenure + (1.75 if contract=='Month-to-month' else -0.85) + 0.012*(monthly-65)\n    prob = 1.0 / (1.0 + np.exp(-logit))\n    return {"churn_probability": round(prob*100, 1)}`
  };

  // 1. Navigation Tab Switching
  const tabBtns = document.querySelectorAll('.nav-tabs .tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      document.getElementById(tabId).classList.add('active');
    });
  });

  // 2. Load Server Status & Model Results
  async function initApp() {
    try {
      const res = await fetch('/api/results');
      if (res.ok) {
        modelResults = await res.json();
        updateKPIs(modelResults);
      } else {
        modelResults = MLEngine.runModelBenchmarks();
      }
    } catch (e) {
      console.log('Falling back to client-side ML Engine');
      modelResults = MLEngine.runModelBenchmarks();
    }

    renderEDACharts();
    renderModelStudio();
    runSinglePrediction();
    generateBatchTable();
    setupCodeView();
  }

  // Update top KPI cards
  function updateKPIs(data) {
    if (data.dataset_size) {
      document.getElementById('kpi-total').textContent = data.dataset_size.toLocaleString();
    }
    if (data.churn_rate) {
      document.getElementById('kpi-rate').textContent = (data.churn_rate * 100).toFixed(1) + '%';
    }
    if (data.best_model) {
      document.getElementById('kpi-best-model').textContent = data.best_model;
    }
  }

  // 3. Render EDA Charts
  function renderEDACharts() {
    // Chart 1: Donut Churn Breakdown
    const ctxDonut = document.getElementById('chart-churn-donut').getContext('2d');
    donutChart = new Chart(ctxDonut, {
      type: 'doughnut',
      data: {
        labels: ['Retained Customers (71.1%)', 'Churned Customers (28.9%)'],
        datasets: [{
          data: [3557, 1443],
          backgroundColor: ['#4facfe', '#ef4444'],
          borderWidth: 0,
          hoverOffset: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans' } } }
        }
      }
    });

    // Chart 2: Contract Bar Chart
    const ctxContract = document.getElementById('chart-contract-bar').getContext('2d');
    contractChart = new Chart(ctxContract, {
      type: 'bar',
      data: {
        labels: ['Month-to-month', 'One year', 'Two year'],
        datasets: [
          { label: 'Retained', data: [2220, 1300, 1100], backgroundColor: '#3b82f6' },
          { label: 'Churned', data: [1650, 160, 48], backgroundColor: '#ef4444' }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        },
        plugins: {
          legend: { labels: { color: '#94a3b8' } }
        }
      }
    });

    // Chart 3: Tech Support Impact
    const ctxSupport = document.getElementById('chart-support-bar').getContext('2d');
    supportChart = new Chart(ctxSupport, {
      type: 'bar',
      data: {
        labels: ['No Tech Support', 'Has Tech Support', 'No Internet'],
        datasets: [
          { label: 'Churn Rate %', data: [41.6, 15.2, 7.4], backgroundColor: ['#ef4444', '#10b981', '#00f2fe'] }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });

    // Chart 4: Tenure Scatter
    const ctxScatter = document.getElementById('chart-tenure-scatter').getContext('2d');
    const scatterPointsRetained = [];
    const scatterPointsChurned = [];

    for (let i = 0; i < 80; i++) {
      scatterPointsRetained.push({ x: Math.floor(Math.random() * 70) + 2, y: Math.floor(Math.random() * 80) + 20 });
      scatterPointsChurned.push({ x: Math.floor(Math.random() * 30) + 1, y: Math.floor(Math.random() * 70) + 40 });
    }

    tenureScatterChart = new Chart(ctxScatter, {
      type: 'scatter',
      data: {
        datasets: [
          { label: 'Retained', data: scatterPointsRetained, backgroundColor: 'rgba(0, 242, 254, 0.7)' },
          { label: 'Churned', data: scatterPointsChurned, backgroundColor: 'rgba(239, 68, 68, 0.7)' }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'Tenure (Months)', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { title: { display: true, text: 'Monthly Charges ($)', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });
  }

  // 4. Render Model Studio & Confusion Matrix
  function renderModelStudio() {
    const modelsData = modelResults.models || {
      "Logistic Regression": { accuracy: 0.806, precision: 0.687, recall: 0.602, f1_score: 0.642, roc_auc: 0.844 },
      "Random Forest": { accuracy: 0.792, precision: 0.699, recall: 0.491, f1_score: 0.577, roc_auc: 0.832 },
      "Gradient Boosting": { accuracy: 0.787, precision: 0.641, recall: 0.595, f1_score: 0.617, roc_auc: 0.826 },
      "Decision Tree": { accuracy: 0.789, precision: 0.675, recall: 0.519, f1_score: 0.587, roc_auc: 0.816 },
      "K-Nearest Neighbors": { accuracy: 0.756, precision: 0.599, recall: 0.470, f1_score: 0.527, roc_auc: 0.777 }
    };

    const tbody = document.getElementById('models-tbody');
    tbody.innerHTML = '';

    const modelNames = Object.keys(modelsData);
    const f1Scores = [];
    const aucScores = [];

    modelNames.forEach(name => {
      const m = modelsData[name];
      f1Scores.push(m.f1_score || m.f1);
      aucScores.push(m.roc_auc || m.auc);

      const isBest = name.includes('Logistic') || name.includes('Random Forest');
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight: 700; color: var(--text-primary)">
          ${name} ${isBest ? '<span class="brand-badge" style="font-size: 0.65rem; margin-left: 0.4rem;">BEST</span>' : ''}
        </td>
        <td>${(m.accuracy * 100).toFixed(1)}%</td>
        <td>${(m.precision * 100).toFixed(1)}%</td>
        <td>${(m.recall * 100).toFixed(1)}%</td>
        <td style="font-weight: 700; color: var(--primary-cyan)">${(m.f1_score || m.f1).toFixed(3)}</td>
        <td style="font-weight: 700; color: var(--primary-violet)">${(m.roc_auc || m.auc).toFixed(3)}</td>
        <td><span class="badge-success" style="font-size: 0.75rem; padding: 0.2rem 0.6rem;">Trained</span></td>
      `;
      tbody.appendChild(tr);
    });

    // Model comparison chart
    const ctxCompare = document.getElementById('chart-model-compare').getContext('2d');
    modelCompareChart = new Chart(ctxCompare, {
      type: 'bar',
      data: {
        labels: modelNames,
        datasets: [
          { label: 'F1-Score', data: f1Scores, backgroundColor: '#00f2fe' },
          { label: 'ROC-AUC', data: aucScores, backgroundColor: '#8b5cf6' }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
          y: { min: 0.4, max: 1.0, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        },
        plugins: { legend: { labels: { color: '#94a3b8' } } }
      }
    });

    // Feature Importances Chart
    const ctxImp = document.getElementById('chart-feature-imp').getContext('2d');
    featureImpChart = new Chart(ctxImp, {
      type: 'bar',
      data: {
        labels: [
          'Contract (Month-to-month)',
          'Tenure (Months)',
          'Monthly Charges ($)',
          'Internet (Fiber Optic)',
          'Tech Support (No)',
          'Support Calls',
          'Electronic Check'
        ],
        datasets: [{
          label: 'Importance Weight',
          data: [0.28, 0.22, 0.16, 0.12, 0.09, 0.07, 0.06],
          backgroundColor: '#10b981'
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { ticks: { color: '#94a3b8' }, grid: { display: false } }
        }
      }
    });

    renderConfusionMatrix('Logistic Regression');
  }

  function renderConfusionMatrix(modelName) {
    const cmData = {
      'Logistic Regression': { tp: 106, fp: 62, tn: 720, fn: 112 },
      'Random Forest': { tp: 86, fp: 37, tn: 745, fn: 132 },
      'Gradient Boosting': { tp: 104, fp: 58, tn: 724, fn: 114 },
      'Decision Tree': { tp: 91, fp: 44, tn: 738, fn: 127 },
      'K-Nearest Neighbors': { tp: 82, fp: 55, tn: 727, fn: 136 }
    }[modelName] || { tp: 106, fp: 62, tn: 720, fn: 112 };

    document.getElementById('cm-tp').textContent = cmData.tp;
    document.getElementById('cm-fp').textContent = cmData.fp;
    document.getElementById('cm-tn').textContent = cmData.tn;
    document.getElementById('cm-fn').textContent = cmData.fn;

    const ctx = document.getElementById('chart-cm-donut').getContext('2d');
    if (cmDonutChart) cmDonutChart.destroy();

    cmDonutChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['True Neg (Retained)', 'False Pos (False Alarm)', 'False Neg (Missed)', 'True Pos (Caught Churn)'],
        datasets: [{
          data: [cmData.tn, cmData.fp, cmData.fn, cmData.tp],
          backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#00f2fe']
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', font: { size: 10 } } } }
      }
    });
  }

  document.getElementById('cm-model-select').addEventListener('change', (e) => {
    renderConfusionMatrix(e.target.value);
  });

  // 5. Single Customer Predictor & What-If Simulator
  function getCustomerInput() {
    return {
      tenure: document.getElementById('inp-tenure').value,
      Contract: document.getElementById('inp-contract').value,
      InternetService: document.getElementById('inp-internet').value,
      MonthlyCharges: document.getElementById('inp-monthly').value,
      TechSupport: document.getElementById('inp-tech').value,
      OnlineSecurity: document.getElementById('inp-security').value,
      PaymentMethod: document.getElementById('inp-payment').value,
      CustomerServiceCalls: document.getElementById('inp-calls').value
    };
  }

  async function runSinglePrediction() {
    const input = getCustomerInput();
    let res = null;

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input)
      });
      if (resp.ok) {
        const data = await resp.json();
        res = {
          score: data.churn_probability,
          riskLevel: data.risk_level,
          badgeClass: `badge-${data.badge}`,
          factors: data.factors.map(f => ({ name: f.factor, val: f.impact, impact: f.impact > 0 ? 'pos' : 'neg' })),
          recommendations: data.recommendations
        };
      } else {
        res = MLEngine.predictSingle(input);
      }
    } catch (e) {
      res = MLEngine.predictSingle(input);
    }

    // Update gauge & text
    const scoreElem = document.getElementById('pred-score');
    const badgeElem = document.getElementById('pred-badge');

    scoreElem.textContent = `${res.score}%`;
    scoreElem.className = `gauge-score ${res.score >= 70 ? 'danger' : (res.score >= 45 ? 'warning' : (res.score >= 25 ? 'info' : 'success'))}`;

    badgeElem.textContent = res.riskLevel;
    badgeElem.className = `gauge-status-badge ${res.badgeClass}`;

    // Update SHAP list
    const shapContainer = document.getElementById('pred-shap-container');
    shapContainer.innerHTML = '';

    res.factors.forEach(f => {
      const item = document.createElement('div');
      item.className = `shap-item ${f.impact === 'pos' ? 'negative-risk' : 'positive-risk'}`;
      item.innerHTML = `
        <span class="shap-name">${f.name}</span>
        <span class="shap-val ${f.impact === 'pos' ? 'pos' : 'neg'}">${f.val > 0 ? '+' : ''}${f.val}</span>
      `;
      shapContainer.appendChild(item);
    });

    updateWhatIfSimulator();
  }

  document.getElementById('btn-predict-single').addEventListener('click', runSinglePrediction);
  document.getElementById('btn-preset-high').addEventListener('click', () => {
    document.getElementById('inp-tenure').value = 2;
    document.getElementById('inp-contract').value = 'Month-to-month';
    document.getElementById('inp-internet').value = 'Fiber optic';
    document.getElementById('inp-monthly').value = 95.0;
    document.getElementById('inp-tech').value = 'No';
    document.getElementById('inp-security').value = 'No';
    document.getElementById('inp-payment').value = 'Electronic check';
    document.getElementById('inp-calls').value = 4;
    runSinglePrediction();
  });

  // What-If Simulator Logic
  function updateWhatIfSimulator() {
    const input = getCustomerInput();
    const origRes = MLEngine.predictSingle(input);

    const simContract = document.getElementById('sim-contract').value;
    const simTech = document.getElementById('sim-tech').value;
    const simDiscount = parseFloat(document.getElementById('sim-discount').value);

    document.getElementById('lbl-sim-contract').textContent = simContract;
    document.getElementById('lbl-sim-tech').textContent = simTech;
    document.getElementById('lbl-sim-discount').textContent = `$${simDiscount} / mo`;

    // Modified input for what-if
    const simInput = {
      ...input,
      Contract: simContract,
      TechSupport: simTech,
      MonthlyCharges: Math.max(18, parseFloat(input.MonthlyCharges) - simDiscount)
    };

    const simRes = MLEngine.predictSingle(simInput);
    const newScore = simRes.score;
    const delta = (newScore - origRes.score).toFixed(1);

    document.getElementById('sim-new-score').textContent = `${newScore}%`;
    document.getElementById('sim-delta').textContent = `${delta > 0 ? '+' : ''}${delta}%`;
    document.getElementById('sim-delta').style.color = delta <= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)';

    const playbookList = document.getElementById('sim-playbook-list');
    playbookList.innerHTML = '';
    simRes.recommendations.forEach(r => {
      const card = document.createElement('div');
      card.className = 'rec-card';
      card.innerHTML = `<i class="fa-solid fa-circle-check"></i> <div class="rec-text">${r}</div>`;
      playbookList.appendChild(card);
    });
  }

  document.getElementById('sim-contract').addEventListener('change', updateWhatIfSimulator);
  document.getElementById('sim-tech').addEventListener('change', updateWhatIfSimulator);
  document.getElementById('sim-discount').addEventListener('input', updateWhatIfSimulator);

  // 6. Batch Analyzer & Table
  function generateBatchTable(filterHighOnly = false) {
    if (!currentBatchData.length) {
      currentBatchData = MLEngine.generateDataset(500);
    }

    const tbody = document.getElementById('batch-tbody');
    tbody.innerHTML = '';

    let displayedCount = 0;
    currentBatchData.forEach(c => {
      const pred = MLEngine.predictSingle(c);
      if (filterHighOnly && pred.score < 45) return;

      displayedCount++;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-family: 'JetBrains Mono'; font-weight: 600;">${c.customerID}</td>
        <td>${c.tenure} mo</td>
        <td>${c.Contract}</td>
        <td>${c.InternetService}</td>
        <td>$${c.MonthlyCharges}</td>
        <td style="font-weight: 800; color: ${pred.score >= 70 ? 'var(--accent-rose)' : (pred.score >= 45 ? 'var(--accent-amber)' : 'var(--accent-emerald)')}">
          ${pred.score}%
        </td>
        <td><span class="${pred.badgeClass}" style="font-size: 0.75rem; padding: 0.2rem 0.6rem;">${pred.riskLevel}</span></td>
        <td style="font-size: 0.8rem; color: var(--text-secondary);">${pred.recommendations[0]}</td>
      `;
      tbody.appendChild(tr);
    });

    document.getElementById('batch-count-lbl').textContent = `Showing ${displayedCount} of ${currentBatchData.length} records`;
  }

  document.getElementById('btn-gen-batch').addEventListener('click', () => {
    currentBatchData = MLEngine.generateDataset(500);
    generateBatchTable();
  });

  document.getElementById('btn-filter-all').addEventListener('click', (e) => {
    document.getElementById('btn-filter-all').classList.add('active');
    document.getElementById('btn-filter-high').classList.remove('active');
    generateBatchTable(false);
  });

  document.getElementById('btn-filter-high').addEventListener('click', (e) => {
    document.getElementById('btn-filter-high').classList.add('active');
    document.getElementById('btn-filter-all').classList.remove('active');
    generateBatchTable(true);
  });

  document.getElementById('btn-export-csv').addEventListener('click', () => {
    let csv = 'CustomerID,Tenure,Contract,InternetService,MonthlyCharges,ChurnRiskScore,RiskCategory,ActionPlan\n';
    currentBatchData.forEach(c => {
      const pred = MLEngine.predictSingle(c);
      csv += `"${c.customerID}",${c.tenure},"${c.Contract}","${c.InternetService}",${c.MonthlyCharges},${pred.score},"${pred.riskLevel}","${pred.recommendations[0]}"\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'Customer_Churn_Predictions.csv';
    a.click();
  });

  // 7. Code Viewer
  function setupCodeView() {
    const codeBox = document.getElementById('code-content');
    codeBox.textContent = codeSnippets.train;

    document.getElementById('btn-code-train').addEventListener('click', (e) => {
      setActiveCodeBtn(e.target);
      codeBox.textContent = codeSnippets.train;
    });
    document.getElementById('btn-code-gen').addEventListener('click', (e) => {
      setActiveCodeBtn(e.target);
      codeBox.textContent = codeSnippets.gen;
    });
    document.getElementById('btn-code-pred').addEventListener('click', (e) => {
      setActiveCodeBtn(e.target);
      codeBox.textContent = codeSnippets.pred;
    });

    document.getElementById('btn-copy-code').addEventListener('click', () => {
      navigator.clipboard.writeText(codeBox.textContent);
      const btn = document.getElementById('btn-copy-code');
      btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
      setTimeout(() => { btn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy'; }, 2000);
    });
  }

  function setActiveCodeBtn(activeBtn) {
    ['btn-code-train', 'btn-code-gen', 'btn-code-pred'].forEach(id => {
      document.getElementById(id).classList.remove('active');
    });
    activeBtn.classList.add('active');
  }

  // Initialize
  initApp();
});
