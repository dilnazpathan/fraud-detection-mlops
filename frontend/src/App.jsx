
import { useEffect, useState } from "react";

const initialForm = {
  step: "1",
  type: "PAYMENT",
  amount: "9839.64",
  oldbalanceOrg: "170136",
  newbalanceOrig: "160296.36",
  oldbalanceDest: "0",
  newbalanceDest: "0",
  isFlaggedFraud: "0",
};

function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [monitoring, setMonitoring] = useState(null);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState("");

  async function loadHistory() {
    setHistoryLoading(true);

    try {
      const response = await fetch("/transactions/?limit=20");

      if (!response.ok) {
        throw new Error("Unable to load transaction history.");
      }

      const data = await response.json();
      setHistory(data.transactions || []);
    } catch (err) {
      console.error("Transaction history error:", err);
    } finally {
      setHistoryLoading(false);
    }
  }

  async function loadMonitoring() {
    try {
      const response = await fetch("/monitoring/summary");

      if (!response.ok) {
        throw new Error("Unable to load monitoring metrics.");
      }

      const data = await response.json();
      setMonitoring(data);
    } catch (err) {
      console.error("Monitoring error:", err);
    }
  }

  useEffect(() => {
    loadHistory();
    loadMonitoring();
  }, []);

  function handleChange(event) {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    const transaction = {
      step: Number(form.step),
      type: form.type,
      amount: Number(form.amount),
      oldbalanceOrg: Number(form.oldbalanceOrg),
      newbalanceOrig: Number(form.newbalanceOrig),
      oldbalanceDest: Number(form.oldbalanceDest),
      newbalanceDest: Number(form.newbalanceDest),
      isFlaggedFraud: Number(form.isFlaggedFraud),
    };

    try {
      const response = await fetch("/prediction/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(transaction),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Prediction failed. Check the transaction details."
        );
      }

      setResult(data);

      await Promise.all([
        loadHistory(),
        loadMonitoring(),
      ]);
    } catch (err) {
      setError(err.message || "Could not connect to the API.");
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    setForm(initialForm);
    setResult(null);
    setError("");
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">F</div>
          <div>
            <h1>FraudWatch</h1>
            <p>Financial transaction intelligence</p>
          </div>
        </div>

        <div className="api-status">
          <span className="status-dot" />
          Fraud Detection API
        </div>
      </header>

      <main className="dashboard">
        <section className="welcome">
          <div>
            <p className="eyebrow">FRAUD DETECTION · MLOPS</p>
            <h2>Transaction Risk Dashboard</h2>
            <p className="welcome-text">
              Analyze financial transactions, review fraud probabilities,
              and monitor predictions saved by your machine-learning model.
            </p>
          </div>

          <div className="model-chip">
            <span className="status-dot" />
            Random Forest · Production model
          </div>
        </section>

        <section className="summary-grid">
          <article className="summary-card">
            <span className="summary-label">Total predictions</span>
            <strong>{monitoring?.total_predictions ?? "—"}</strong>
            <p>Records saved in SQLite</p>
          </article>

          <article className="summary-card">
            <span className="summary-label">Predicted fraud</span>
            <strong>{monitoring?.predicted_fraud ?? "—"}</strong>
            <p>Transactions classified as fraud</p>
          </article>

          <article className="summary-card">
            <span className="summary-label">Predicted fraud rate</span>
            <strong>
              {monitoring
                ? `${Number(monitoring.predicted_fraud_rate_percent).toFixed(2)}%`
                : "—"}
            </strong>
            <p>Share of saved predictions classified as fraud</p>
          </article>

          <article className="summary-card">
            <span className="summary-label">Average fraud probability</span>
            <strong>
              {monitoring
                ? `${(Number(monitoring.average_fraud_probability) * 100).toFixed(2)}%`
                : "—"}
            </strong>
            <p>Average risk probability across saved predictions</p>
          </article>
        </section>

        <div className="content-grid">
          <section className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">NEW ANALYSIS</p>
                <h3>Check a transaction</h3>
              </div>
              <span className="panel-number">01</span>
            </div>

            <form onSubmit={handleSubmit}>
              <div className="form-grid">
                <label>
                  Transaction step
                  <input
                    name="step"
                    type="number"
                    min="1"
                    step="1"
                    value={form.step}
                    onChange={handleChange}
                    required
                  />
                </label>

                <label>
                  Transaction type
                  <select
                    name="type"
                    value={form.type}
                    onChange={handleChange}
                  >
                    <option value="PAYMENT">Payment</option>
                    <option value="TRANSFER">Transfer</option>
                    <option value="CASH_OUT">Cash out</option>
                    <option value="CASH_IN">Cash in</option>
                    <option value="DEBIT">Debit</option>
                  </select>
                </label>

                <label>
                  Transaction amount
                  <input
                    name="amount"
                    type="number"
                    min="0"
                    step="any"
                    value={form.amount}
                    onChange={handleChange}
                    required
                  />
                </label>

                <label>
                  Sender old balance
                  <input
                    name="oldbalanceOrg"
                    type="number"
                    min="0"
                    step="any"
                    value={form.oldbalanceOrg}
                    onChange={handleChange}
                    required
                  />
                </label>

                <label>
                  Sender new balance
                  <input
                    name="newbalanceOrig"
                    type="number"
                    min="0"
                    step="any"
                    value={form.newbalanceOrig}
                    onChange={handleChange}
                    required
                  />
                </label>

                <label>
                  Receiver old balance
                  <input
                    name="oldbalanceDest"
                    type="number"
                    min="0"
                    step="any"
                    value={form.oldbalanceDest}
                    onChange={handleChange}
                    required
                  />
                </label>

                <label>
                  Receiver new balance
                  <input
                    name="newbalanceDest"
                    type="number"
                    min="0"
                    step="any"
                    value={form.newbalanceDest}
                    onChange={handleChange}
                    required
                  />
                </label>

                <label>
                  Previously flagged?
                  <select
                    name="isFlaggedFraud"
                    value={form.isFlaggedFraud}
                    onChange={handleChange}
                  >
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </label>
              </div>

              {error && (
                <div className="error-message" role="alert">
                  {error}
                </div>
              )}

              <div className="form-actions">
                <button type="submit" disabled={loading}>
                  {loading ? "Analyzing transaction..." : "Analyze transaction"}
                </button>

                <button
                  className="secondary-button"
                  type="button"
                  onClick={resetForm}
                >
                  Reset form
                </button>
              </div>
            </form>
          </section>

          <section className="panel result-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">MODEL OUTPUT</p>
                <h3>Prediction result</h3>
              </div>
              <span className="panel-number">02</span>
            </div>

            {!result ? (
              <div className="empty-result">
                <div className="empty-icon">◎</div>
                <h4>Awaiting transaction</h4>
                <p>
                  Submit the form to see the predicted class and estimated
                  fraud probability.
                </p>
              </div>
            ) : (
              <div className="prediction-result">
                <span
                  className={`prediction-badge ${
                    result.prediction === "FRAUD"
                      ? "fraud"
                      : "legitimate"
                  }`}
                >
                  {result.prediction === "FRAUD"
                    ? "Potential fraud"
                    : "Legitimate"}
                </span>

                <p className="probability-label">
                  Estimated fraud probability
                </p>

                <strong className="probability-value">
                  {Number(result.fraud_probability_percent).toFixed(2)}%
                </strong>

                <div
                  className="probability-track"
                  role="progressbar"
                  aria-valuemin="0"
                  aria-valuemax="100"
                  aria-valuenow={Number(result.fraud_probability_percent)}
                  aria-label="Fraud probability"
                >
                  <div
                    className={`probability-fill ${
                      result.prediction === "FRAUD" ? "fraud" : ""
                    }`}
                    style={{
                      width: `${Math.min(
                        100,
                        Math.max(
                          0,
                          Number(result.fraud_probability_percent)
                        )
                      )}%`,
                    }}
                  />
                </div>

                <div className="result-detail">
                  <span>Applied threshold</span>
                  <strong>{Number(result.threshold).toFixed(6)}</strong>
                </div>

                <div className="result-detail">
                  <span>Saved record ID</span>
                  <strong>{result.record_id ?? "Not provided"}</strong>
                </div>

                <p className="result-note">
                  This is a model-generated risk estimate, not definitive
                  proof that a transaction is fraudulent.
                </p>
              </div>
            )}
          </section>
        </div>

        <section className="panel history-panel">
          <div className="panel-heading history-heading">
            <div>
              <p className="eyebrow">DATABASE RECORDS</p>
              <h3>Recent predictions</h3>
              <p className="welcome-text">
                Latest transactions saved by the prediction API.
              </p>
            </div>

            <button
              className="secondary-button refresh-button"
              type="button"
              onClick={() => Promise.all([loadHistory(), loadMonitoring()])}
              disabled={historyLoading}
            >
              {historyLoading ? "Refreshing..." : "Refresh history"}
            </button>
          </div>

          {historyLoading && history.length === 0 ? (
            <div className="empty-history">Loading saved predictions...</div>
          ) : history.length === 0 ? (
            <div className="empty-history">
              No saved predictions found yet.
            </div>
          ) : (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Record ID</th>
                    <th>Type</th>
                    <th>Amount</th>
                    <th>Prediction</th>
                    <th>Fraud probability</th>
                    <th>Created at</th>
                  </tr>
                </thead>

                <tbody>
                  {history.map((item) => (
                    <tr key={item.record_id}>
                      <td>#{item.record_id}</td>
                      <td>{item.type}</td>
                      <td>{Number(item.amount).toLocaleString()}</td>
                      <td>
                        <span
                          className={`table-badge ${
                            item.prediction === "FRAUD"
                              ? "fraud"
                              : "legitimate"
                          }`}
                        >
                          {item.prediction}
                        </span>
                      </td>
                      <td>
                        {(Number(item.fraud_probability) * 100).toFixed(2)}%
                      </td>
                      <td>
                        {item.created_at
                          ? new Date(item.created_at).toLocaleString()
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <footer className="footer">
          FraudWatch · Fraud Detection MLOps · Demonstration and evaluation
        </footer>
      </main>
    </div>
  );
}

export default App;
