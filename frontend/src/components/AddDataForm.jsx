import { useState, useEffect } from "react";
import { addDataRecord } from "../api";

const dayAfter = (isoDate) => {
  const d = new Date(isoDate + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + 1);
  return d.toISOString().slice(0, 10);
};

export default function AddDataForm({ storeNbr, lastDataDate, onAdded }) {
  const [form, setForm] = useState({
    date: lastDataDate ? dayAfter(lastDataDate) : "",
    sales: "",
    onpromotion: "0",
  });
  const [status, setStatus] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (lastDataDate) {
      setForm((f) => ({ ...f, date: f.date || dayAfter(lastDataDate) }));
    }
  }, [lastDataDate]);

  const update = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setStatus(null);
    try {
      await addDataRecord({
        date: form.date,
        store_nbr: storeNbr,
        sales: parseFloat(form.sales),
        onpromotion: parseInt(form.onpromotion || "0", 10),
      });
      setStatus({ type: "ok", msg: "Record added. Forecast will use this on next refresh." });
      setForm((f) => ({ ...f, sales: "" }));
      onAdded?.();
    } catch (err) {
      setStatus({ type: "err", msg: err.response?.data?.detail || "Failed to add record." });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-grid">
        <div className="field">
          <label>Date {lastDataDate && <span style={{ fontWeight: 400, color: "#5d7369" }}>(latest on file: {lastDataDate})</span>}</label>
          <input type="date" value={form.date} onChange={update("date")} required />
        </div>
        <div className="field">
          <label>Store</label>
          <input type="text" value={`#${storeNbr}`} disabled />
        </div>
        <div className="field">
          <label>Daily Sales ($)</label>
          <input
            type="number"
            min="0"
            step="0.01"
            placeholder="e.g. 4820.50"
            value={form.sales}
            onChange={update("sales")}
            required
          />
        </div>
        <div className="field">
          <label>Items on Promotion</label>
          <input
            type="number"
            min="0"
            value={form.onpromotion}
            onChange={update("onpromotion")}
          />
        </div>
      </div>

      <div style={{ marginTop: 16, display: "flex", gap: 10 }}>
        <button className="btn" type="submit" disabled={submitting}>
          {submitting ? "Adding..." : "Add Record"}
        </button>
      </div>

      {status && <div className={`status-msg ${status.type}`}>{status.msg}</div>}
    </form>
  );
}
